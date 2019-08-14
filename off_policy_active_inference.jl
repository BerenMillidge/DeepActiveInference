using OpenAIGym
import Random
using Statistics
using Flux
using StatsBase

const MEM_SIZE = 100000
const BATCH_SIZE = 200
const STATE_SIZE = 4
const ACTION_SIZE = 2

mutable struct History
    nS::Int
    nA::Int
    γ::Float64
    states::Vector{Float64}
    actions::Vector{Int}
    rewards::Vector{Float64}
end

History(nS, nA, γ) = History(nS, nA, γ, zeros(0),zeros(Int, 0),zeros(0))

memory = []
function remember(state, action, reward, next_state, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, (state, action, reward, next_state, done))
end

value_loss(x, y) = Flux.mse(x,y)

ploss(ps, vs) = -mean(sum(ps .* logsoftmax(vs), dims=1))

function replay(opt_v, opt_p, policynet, valuenet, deep_value_net)
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Matrix{Float32}(undef,STATE_SIZE, batch_size)
  y = Matrix{Float32}(undef,ACTION_SIZE, batch_size)
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
  qhats = valuenet(x)
  ps = softmax(policynet(x))

  Flux.train!(value_loss,Flux.params(valuenet),[(qhats, y)], opt_v)
  #println(value_loss(qhats, y))
  Flux.train!(ploss, Flux.params(policynet), [(ps, qhats.data)], opt_p)
  return value_loss(qhats, y), ploss(ps, qhats.data)
end
#basenet = Dense(4,100, Flux.relu)


function sample_action(probs)
    @assert size(probs, 2) == 1
    cprobs = cumsum(probs, dims=1)
    sampled = cprobs .> rand()
    return mapslices(argmax, sampled, dims=1)[1] 
end

function mean_ac_loss(history)
    nS, nA = history.nS, history.nA
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    p = softmax(policynet(states))
    V = valuenet(states)
    ploss = -mean(sum(p .* logsoftmax(V.data), dims=1))
    #println("ploss: $ploss")
    return ploss
end

mean_mean_ac_loss(histories) = mean(mean_ac_loss.(histories)

L2Reg(x) = mean(x .* x)
function main(
    hidden = [100], # width inner layers
    lr = 1e-2,
    γ = 0.99, #discount rate
    episodes = 15000,
    infotime = 50)
    env = GymEnv("CartPole-v1")
    seed = -1
    seed > 0 && (Random.seed!(seed); Gym.seed!(env, seed))

    valuenet = Chain(Dense(4,100, Flux.relu),Dense(100,2))
    policynet = Chain(Dense(4,100, Flux.relu), Dense(100,2))
    deep_value_net = deepcopy(valuenet)
    opt_p=ADAM(0.001)
    opt_v = ADAM(0.001)
    nS, nA = 4, 2
    avgreward = 0
    histories = []
    rewards = []
    plosses = []
    vlosses = []
    for episode=1:episodes
        state = reset!(env)
        episode_rewards = 0
        history = History(nS, nA, γ)
        for t=1:10000
            #println("in inner loop!")
            #p, V = predict(state)
            p = policynet(state)
            p = softmax(p)
            #println("$(typeof(p)), $(size(p))")
            #println("action probs: $(p.data)")
            action = sample_action(p.data)

            reward, next_state = step!(env, action-1)
            append!(history.states, state)
            push!(history.actions, action)
            push!(history.rewards, reward)
            done = env.done
            remember(state, action, reward, next_state, done)
            state = next_state
            episode_rewards += reward

            #episode % infotime == 0 && render(env)
            done && break # this breaks it after every episode!
        end
        push!(histories, history)
        avgreward = 0.1 * episode_rewards + avgreward * 0.9
        if episode % infotime == 0
            println("(episode:$episode, avgreward:$avgreward)")
            close(env)
        end
        #if episode % 5 == 0
        #    Flux.train!(mean_mean_ac_loss, Flux.params(valuenet, policynet), [[histories]], opt_p)
        #    histories = []
        #end
        if episode % 50 == 0
            deep_value_net = deepcopy(valuenet)
        end
        #Flux.train!(mean_ac_loss, Flux.params(valuenet, policynet), [[history]], opt_p)
        vloss, ploss = replay(opt_v, opt_p,policynet, valuenet,deep_value_net)
        push!(rewards, episode_rewards)
        push!(vlosses, vloss.data)
        push!(plosses, ploss.data)
    end
    return rewards, vlosses, plosses
end
using BSON
function save_results()
    rs = []
    pls = []
    vls = []
    for i in 1:20
      ep_rewards, vlosses, plosses = main()
      push!(rs, ep_rewards)
      push!(pls, plosses)
      push!(vls, vlosses)
      BSON.bson("results/off_policy_active_inference.bson", a=[rs,pls,vls])
      println("save successful!")
    end
end
save_results()


