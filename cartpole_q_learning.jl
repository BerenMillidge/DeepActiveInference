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

function remember(memory, state, action, reward, next_state, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, (state, action, reward, next_state, done))
end

value_loss(x, y) = Flux.mse(x,y)

function replay(opt_v, valuenet, deep_value_net,memory)
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
  Flux.train!(value_loss,Flux.params(valuenet),[(qhats, y)], opt_v)
  #println(value_loss(qhats, y))
  return value_loss(qhats, y)

end

function replay_expectation(opt_v, valuenet, deep_value_net,memory, policynet)
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Matrix{Float32}(undef,STATE_SIZE, batch_size)
  y = Matrix{Float32}(undef,ACTION_SIZE, batch_size)
  for (iter, (state, action, reward, next_state, done)) in enumerate(minibatch)
    target = reward
    if !done
      target += 0.99f0 * sum(softmax(policynet(next_state)) .* deep_value_net(next_state)) 
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

function sample_action(probs)
    return Flux.argmax(probs) 
end

function main(
    γ = 0.99, #discount rate
    episodes = 15000,
    render = true,
    infotime = 50)
    env = GymEnv("CartPole-v1")
    seed = -1
    seed > 0 && (Random.seed!(seed); Gym.seed!(env, seed))

    valuenet = Chain(Dense(STATE_SIZE,100, Flux.relu),Dense(100,ACTION_SIZE))
    deep_value_net = deepcopy(valuenet)
    opt_p=ADAM(0.001)
    opt_v = ADAM(0.001)
    nS, nA = STATE_SIZE, ACTION_SIZE
    avgreward = 0
    histories = []
    ep_rewards = []
    vlosses = []
    plosses = []
    tlosses = []
    memory = []
    for episode=1:episodes
        state = reset!(env)
        episode_rewards = 0
        history = History(nS, nA, γ)
        for t=1:10000
            p = valuenet(state)
            action = sample_action(p.data)

            reward, next_state = step!(env, action-1)
            append!(history.states, state)
            push!(history.actions, action)
            push!(history.rewards, reward)
            done = env.done
            remember(memory, state, action, reward, next_state, done)
            state = next_state
            episode_rewards += reward

            #episode % infotime == 0 && render && Gym.render(env)
            done && break
        end
        push!(histories, history)
        avgreward = 0.1 * episode_rewards + avgreward * 0.9
        if episode % infotime == 0
            println("(episode:$episode, avgreward:$avgreward)")
            close(env)
        end
        if episode % 50 == 0
            deep_value_net = deepcopy(valuenet)
        end
        #Flux.train!(mean_ac_loss, Flux.params(valuenet, policynet), [[history]], opt_p)
        vloss = replay(opt_v, valuenet, deep_value_net, memory)
    
        #println("tloss: $tloss")
        push!(ep_rewards, episode_rewards)
        push!(vlosses, vloss.data)
    end
    return ep_rewards, vlosses
end
using BSON
function save_results()
    rs = []
    vls = []
    for i in 1:20
      ep_rewards, vlosses = main()
      push!(rs, ep_rewards)
      push!(vls, vlosses)
      BSON.bson("results/q_learning_baseline.bson", a=[rs,vls])
      println("save successful!")
    end
end

save_results()

