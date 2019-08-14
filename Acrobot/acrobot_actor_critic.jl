
using OpenAIGym
import AutoGrad
import Random
using Statistics
using Flux
using StatsBase

const STATE_SIZE = 6
const ACTION_SIZE =3
const MEM_SIZE = 100000
const BATCH_SIZE = 200

mutable struct History
    nS::Int
    nA::Int
    γ::Float64
    states::Vector{Float64}
    actions::Vector{Int}
    rewards::Vector{Float64}
end
History(nS, nA, γ) = History(nS, nA, γ, zeros(0),zeros(Int, 0),zeros(0))

function predict(s, policynet,valuenet)
    #println(size(s))
    #baseout = basenet(s)
    v = valuenet(s)
    p = policynet(s)
    return p, v
end
function remember(memory, state, action, reward, next_state, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, (state, action, reward, next_state, done))
end

function sample_action(probs)
    @assert size(probs, 2) == 1
    cprobs = cumsum(probs, dims=1)
    sampled = cprobs .> rand()
    return mapslices(argmax, sampled, dims=1)[1] 
end

value_loss(x,y) = Flux.mse(x,y)
function replay(opt, valuenet, deep_value_net, memory)
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Array{Float32, 2}(undef, STATE_SIZE, batch_size)
  y = Array{Float32,2}(undef, ACTION_SIZE, batch_size)
  for (iter, (state, action, reward, next_state, done)) in enumerate(minibatch)
    target = reward
    if !done
      target += 0.99f0 * maximum(deep_value_net(next_state |> gpu).data)
    end

    target_f = valuenet(state |> gpu).data
    target_f[action] = target

    x[:, iter] .= state
    y[:, iter] .= target_f
  end
  qhat = valuenet(x)

  Flux.train!(value_loss,Flux.params(valuenet), [(qhat, y)], opt)
  return value_loss(qhat,y)
end
function replay_expectation(opt_v, valuenet, deep_value_net,memory, policynet)
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Matrix{Float32}(undef,STATE_SIZE, batch_size)
  y = Matrix{Float32}(undef,ACTION_SIZE, batch_size)
  for (iter, (state, action, reward, next_state, done)) in enumerate(minibatch)
    target = reward
    if !done
      target += 0.99f0 * sum(softmax(policynet(next_state)) .* deep_value_net(next_state)).data 
    end

    target_f = valuenet(state |> gpu).data
    target_f[action] = target

    x[:, iter] .= state
    y[:, iter] .= target_f
  end
  qhats = valuenet(x)
  Flux.train!(value_loss,Flux.params(valuenet),[(qhats, y)], opt_v)
  #println(value_loss(qhats, y))
  return value_loss(qhats, y)

end

function policy_loss(lp,V)
    l = mean(lp .* V)
    return l
end

function loss(history,opt_p, opt_v, policynet, valuenet)
    #println("$(typeof(history)), $history")
    nS, nA = history.nS, history.nA
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    p,V = predict(states, policynet, valuenet)
    inds = history.actions + nA*(0:M-1) 
    V = V[inds].data
    lp = logsoftmax(p)[inds] 
    Flux.train!(policy_loss, Flux.params(policynet), [[lp,V]], opt_p)
    return policy_loss(lp,V)
end

function main(
    γ = 0.99, #discount rate
    episodes = 15000,
    render = true,
    infotime = 50)
    env = GymEnv("Acrobot-v1")
    seed = -1
    seed > 0 && (Random.seed!(seed); seed!(env, seed))
    opt_p=ADAM(0.005)
    opt_v = ADAM(0.005)
    nS, nA = STATE_SIZE, ACTION_SIZE
    #opt = [Adam(lr=lr) for _=1:length(w)]
    avgreward = 0
    ep_rewards = []
    plosses = []
    vlosses = []
    valuenet = Chain(Dense(STATE_SIZE, 100,Flux.relu), Dense(100,ACTION_SIZE))
    policynet = Chain(Dense(STATE_SIZE, 100, Flux.relu), Dense(100,ACTION_SIZE))
    deep_value_net = deepcopy(valuenet)
    memory = []
    for episode=1:episodes
        state = reset!(env)
        episode_rewards = 0
        history = History(nS, nA, γ)
        for t=1:50000
            #println("in inner loop!")
            p, V = predict(state, policynet, valuenet)
            p = softmax(p)
            #println("$(typeof(p)), $(size(p))")
            action = sample_action(p.data)

            reward, next_state = step!(env, action-1)
            done = env.done
            append!(history.states, state)
            push!(history.actions, action)
            push!(history.rewards, reward)
            state = next_state
            episode_rewards += reward
            remember(memory, state,action, reward, next_state, done)

            #episode % infotime == 0 && render && Gym.render(env)
            done && break # this breaks it after every episode!
        end
        #println("episode:$episode, episode reward:$episode_rewards")
        avgreward = 0.1 * episode_rewards + avgreward * 0.9
        if episode % infotime == 0
            println("episode:$episode, avgreward:$avgreward")
            close(env)
        end
        if episode % 50 == 0
            deep_value_net = deepcopy(valuenet)
        end
        ploss = loss(history,opt_p, opt_v, policynet, valuenet)
        #vloss = replay(opt_v, valuenet,deep_value_net,memory)
        vloss = replay(opt_v, valuenet, deep_value_net, memory)
        push!(ep_rewards, episode_rewards)
        push!(plosses, ploss.data)
        push!(vlosses, vloss.data)
    end
    return ep_rewards, plosses, vlosses
end
using BSON
function save_results()
    rs = []
    pls = []
    vls = []
    for i in 1:10
      ep_rewards, plosses, vlosses = main()
      push!(rs, ep_rewards)
      push!(pls, plosses)
      push!(vls, vlosses)
      BSON.bson("results/acrobot_actor_critic_positive.bson", a=[rs,pls,vls])
      println("save successful!")
    end

end

save_results()
