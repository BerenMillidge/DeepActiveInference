using OpenAIGym
import Random
using Statistics
using Flux
using StatsBase
using Flux.Tracker

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
struct memory_item
    state::Array{Float32,1}
    action::Int32
    reward::Int32
    next_state::Array{Float32,1}
    done::Bool
end

History(nS, nA, γ) = History(nS, nA, γ, zeros(0),zeros(Int, 0),zeros(0))
precis = Flux.Tracker.param(1f0)

function remember(memory,item::memory_item)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, item)
end
value_loss(x, y) = Flux.mse(x,y)

function replay(memory,opt_v,valuenet, deep_value_net)
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Matrix{Float32}(undef,STATE_SIZE, batch_size)
  y = Matrix{Float32}(undef,ACTION_SIZE, batch_size)
  for (iter, item) in enumerate(minibatch)
    target = item.reward
    if !item.done
      target += 0.99f0 * maximum(deep_value_net(item.next_state).data)
    end

    target_f = valuenet(item.state).data
    target_f[item.action] = target

    x[:, iter] .= item.state
    y[:, iter] .= target_f
  end
  qhats = valuenet(x)
  Flux.train!(value_loss,Flux.params(valuenet),[(qhats, y)], opt_v)
  #println(value_loss(qhats, y))
  return value_loss(qhats, y)

end
#basenet = Dense(4,100, Flux.relu)

function sample_action(probs)
    @assert size(probs, 2) == 1
    cprobs = cumsum(probs, dims=1)
    sampled = cprobs .> rand()
    return mapslices(argmax, sampled, dims=1)[1]
end

function mean_ac_loss(history, policynet, valuenet)
    nS, nA = history.nS, history.nA
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    p = softmax(policynet(states .* precis))
    V = valuenet(states)
    ploss = -mean(sum(p .* logsoftmax(V.data), dims=1))
    entloss = mean(p.*log.(p))
    #println("ploss: $ploss, entloss: $entloss")
    return ploss + entloss
end

mean_mean_ac_loss(histories, policynet, valuenet) = mean([mean_ac_loss(hist, policynet,valuenet) for hist in histories])

function main(
    hidden = [100], # width inner layers
    lr = 1e-2,
    γ = 0.99, #discount rate
    episodes = 15000,
    render = true,
    infotime = 50)
    env = GymEnv("CartPole-v1")
    seed = -1
    seed > 0 && (Random.seed!(seed); Gym.seed!(env, seed))

    global deep_value_net, precis
    valuenet = Chain(Dense(4,100, Flux.relu),Dense(100,2))
    policynet = Chain(Dense(4,100, Flux.relu), Dense(100,2))
    deep_value_net = deepcopy(valuenet)
    opt_p=ADAM(0.001)
    opt_v = ADAM(0.001)
    opt_precis = ADAM(0.001)
    nS, nA = 4, 2
    avgreward = 0
    histories = []
    ep_rewards = []
    vlosses = []
    plosses = []
    tlosses = []
    precisions = []
    memory = Array{memory_item,1}()
    for episode=1:episodes
        state = reset!(env)
        episode_rewards = 0
        history = History(nS, nA, γ)
        for t=1:10000
            p = policynet(state)
            p = softmax(p .* precis)
            action = sample_action(p.data)

            reward, next_state = step!(env, action-1)
            append!(history.states, state)
            push!(history.actions, action)
            push!(history.rewards, reward)
            done = env.done
            remember(memory,memory_item(state, action, reward, next_state, done))
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
        #Flux.train!(precis_loss,Flux.params(precis), [[history]], opt_precis)
        if episode % 5 == 0
            Flux.train!(mean_mean_ac_loss, Flux.params(policynet), [[histories, policynet,valuenet]], opt_p)
            grad = Tracker.grad(precis)
            #println(grad)
            precis = precis.data
            precis -= 0.1f0 * grad
            #println(precis)
            precis = Flux.Tracker.param(precis)
     
            #Flux.train!(mean_mean_ac_loss, Flux.params(precis), [[histories]], opt_precis)
            histories = []
        end
        if episode % 50 == 0
            deep_value_net = deepcopy(valuenet)
        end
        #Flux.train!(mean_ac_loss, Flux.params(valuenet, policynet), [[history]], opt_p)
        vloss = replay(memory,opt_v, valuenet, deep_value_net)
        #println("tloss: $tloss")
        push!(ep_rewards, episode_rewards)
        push!(plosses, mean_ac_loss(history, policynet,valuenet).data)
        push!(vlosses, vloss.data)
        push!(precisions, precis.data)
    end
    return ep_rewards, plosses, vlosses, precisions
end

using BSON
function save_results()
    rs = []
    pls = []
    vls = []
    prs = []
    for i in 1:20
      ep_rewards, plosses, vlosses, precisions = main()
      push!(rs, ep_rewards)
      push!(pls, plosses)
      push!(vls, vlosses)
      push!(prs, precisions)
      BSON.bson("results/active_inference_learned_precision.bson", a=[rs,pls,vls, prs])
      println("save successful!")
    end
end
#env = GymEnv("CartPole-v1")
#save_results()
#ep_rewards, precisions = main()
#using Plots
#plot(precisions)
#plot(ep_rewards)
save_results()
