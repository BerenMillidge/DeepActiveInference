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
struct memory_item
    state::Array{Float32,1}
    action::Int32
    reward::Int32
    next_state::Array{Float32,1}
    done::Bool
end

History(nS, nA, γ) = History(nS, nA, γ, zeros(0),zeros(Int, 0),zeros(0))

function remember(memory,item::memory_item)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, item)
end
value_loss(x, y) = Flux.mse(x,y)

function replay(opt_v::ADAM, opt_t::ADAM, memory, valuenet, deep_value_net, Tmodel)
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Matrix{Float32}(undef,STATE_SIZE, batch_size)
  #sdash = Matrix{Float32}(undef, STATE_SIZE, batch_size)
  y = Matrix{Float32}(undef,ACTION_SIZE, batch_size)
  states = Array{Array{Float32,1},1}()
  sdashes = Array{Array{Float32,1},1}()
  actions = Array{Array{Float32,1},1}()
  for (iter, item) in enumerate(minibatch)
    target = item.reward
    bonuses = zeros(ACTION_SIZE)
    action = onehotAction(item.action)
    for i in 1:ACTION_SIZE
        a = onehotAction(i)
        bonus::Float32 = sum((item.next_state .- Tmodel(vcat(item.state,a)).data)::Array{Float32,1} .^ 2)
        bonuses[i] = bonus
    end
    #println(ep_bonus)
    #println("bonuses: $bonuses")
    if !item.done
      target += 0.99f0 * maximum(deep_value_net(item.next_state).data)
    end
    target_f = valuenet(item.state).data
    target_f[item.action] = target
    #println("target_f $(size(target_f)), ep_bonus: $(size(ep_bonus))")
    target_f .+= bonuses
    x[:, iter] .= item.state
    y[:, iter] .= target_f
    #sdash[:,iter] .= item.next_state
    push!(states, item.state)
    push!(sdashes, item.next_state)
    push!(actions, action)
  end
  qhats = valuenet(x)
  Flux.train!(value_loss,Flux.params(valuenet), [(qhats, y)], opt_v)
  Flux.train!(Tloss, Flux.params(Tmodel), [(states,actions, sdashes, Tmodel)], opt_t)
  Flux.truncate!(Tmodel)
  return value_loss(qhats, y), Tloss(states, actions, sdashes,Tmodel)
end
function onehotAction(a)
    arr = zeros(ACTION_SIZE)
    arr[a] = 1
    return arr
end

function Tloss(states::Array{Array{Float32,1},1},actions::Array{Array{Float32,1}},sdashes::Array{Array{Float32,1},1}, Tmodel)
    sdashhats = Tmodel.(vcat.(states,actions))
    Tloss = sum(Flux.mse.(sdashhats, sdashes))
    return Tloss
end

function sample_action(probs)
    @assert size(probs, 2) == 1
    cprobs = cumsum(probs, dims=1)
    sampled = cprobs .> rand()
    return mapslices(argmax, sampled, dims=1)[1] # wtf is this?
end

function mean_ac_loss(history, policynet, valuenet)
    nS, nA = history.nS, history.nA
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    p = softmax(policynet(states))
    V = valuenet(states)
    ploss = -mean(sum(p .* logsoftmax(V.data), dims=1))
    entloss = mean(p.*log.(p))
    #println("ploss: $ploss, entloss: $entloss")
    return ploss + entloss
end

mean_mean_ac_loss(histories, policynet, valuenet) = mean([mean_ac_loss(hist, policynet, valuenet) for hist in histories])

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

    valuenet = Chain(Dense(STATE_SIZE,100, Flux.relu),Dense(100,ACTION_SIZE))
    policynet = Chain(Dense(STATE_SIZE,100, Flux.relu), Dense(100,ACTION_SIZE))
    deep_value_net = deepcopy(valuenet)
    Tmodel = Chain(Dense(STATE_SIZE + ACTION_SIZE,100, Flux.relu), Dense(100,STATE_SIZE))

    opt_p=ADAM(0.001)
    opt_v = ADAM(0.001)
    opt_t = ADAM(0.001)
    nS, nA = 4, 2
    avgreward = 0
    histories = []
    ep_rewards = []
    vlosses = []
    plosses = []
    tlosses = []
    memory = Array{memory_item,1}()
    for episode=1:episodes
        state = reset!(env)
        episode_rewards = 0
        history = History(nS, nA, γ)
        for t=1:10000
            p = policynet(state)
            p = softmax(p)
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
        if episode % 5 == 0
            Flux.train!(mean_mean_ac_loss, Flux.params(valuenet, policynet), [[histories, policynet, valuenet]], opt_p)
            histories = []
        end
        if episode % 50 == 0
            deep_value_net = deepcopy(valuenet)
        end
        Flux.train!(mean_ac_loss, Flux.params(valuenet, policynet), [[history, policynet, valuenet]], opt_p)
        vloss, tloss = replay(opt_t,opt_v, memory, valuenet, deep_value_net, Tmodel)
        #println("tloss: $tloss")
        push!(ep_rewards, episode_rewards)
        push!(plosses, mean_ac_loss(history, policynet, valuenet).data)
        push!(vlosses, vloss.data)
        push!(tlosses, tloss.data)
    end
    return ep_rewards, plosses, vlosses, tlosses
end
using BSON
function save_results()
    rs = []
    pls = []
    vls = []
    tls = []
    for i in 1:20
      ep_rewards, plosses, vlosses,tlosses = main()
      push!(rs, ep_rewards)
      push!(pls, plosses)
      push!(vls, vlosses)
      push!(tls, tlosses)
      BSON.bson("results/action_specific_Tmodel.bson", a=[rs,pls,vls,tls])
      println("save successful!")
    end
end

save_results()
