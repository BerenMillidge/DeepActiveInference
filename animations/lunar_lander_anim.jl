
using OpenAIGym
import Random
using Statistics
using Flux
using StatsBase

const MEM_SIZE = 100000
const BATCH_SIZE = 200
const STATE_SIZE = 8
const ACTION_SIZE = 4

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
    reward::Float32
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

function discount(rewards, γ)
    R = similar(rewards)
    R[end] = rewards[end]
    for k = length(rewards)-1:-1:1
        R[k] = γ * R[k+1] + rewards[k]
    end
    return R
    #return (R .- mean(R)) ./ (std(R) + 1e-10) #speeds up training a lot
end

value_loss(x, y) = Flux.mse(x,y)

function replay(memory,opt_v, valuenet, deep_value_net)
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
function replay_expectation(memory,opt_v, valuenet, deep_value_net, policynet)
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Matrix{Float32}(undef,STATE_SIZE, batch_size)
  y = Matrix{Float32}(undef,ACTION_SIZE, batch_size)
  for (iter, item) in enumerate(minibatch)
    target = item.reward
    if !item.done
      target += 0.99f0 * sum(softmax(policynet(item.next_state)) .* deep_value_net(item.next_state)).data
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
    p = softmax(policynet(states))
    V = valuenet(states)
    ploss = -mean(sum(p .* logsoftmax(V.data), dims=1))
    entloss = mean(p.*log.(p))
    #println("ploss: $ploss, entloss: $entloss")
    return ploss + entloss
end

mean_mean_ac_loss(histories, policynet, valuenet) = mean([mean_ac_loss(hist, policynet, valuenet) for hist in histories])

using PyCall
using Plots, Images, ImageView
np = pyimport("numpy")
function main(
    γ = 0.99, #discount rate
    episodes = 15000,
    render = true,
    infotime = 50)
    env = GymEnv("LunarLander-v2")
    seed = -1
    seed > 0 && (Random.seed!(seed); Gym.seed!(env, seed))

    valuenet = Chain(Dense(STATE_SIZE,100, Flux.relu),Dense(100,ACTION_SIZE))
    policynet = Chain(Dense(STATE_SIZE,100, Flux.relu), Dense(100,ACTION_SIZE))
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
    memory = Array{memory_item,1}()
    num_anim = 0
    pyenv = env.pyenv
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
            #println("action: $action")

            reward, next_state = step!(env, action-1)
            #println("reward: $reward")
            append!(history.states, state)
            push!(history.actions, action)
            push!(history.rewards, reward)
            done = env.done
            remember(memory,memory_item(state, action, reward, next_state, done))
            state = next_state
            episode_rewards += reward

            #episode % infotime == 0 && render && Gym.render(env)
            done && break # this breaks it after every episode!
        end
        if avgreward > -50 && episode > 100 && num_anim < 20
            state = reset!(env)
            frames = []
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
                #remember(state, action, reward, next_state, done)
                state = next_state
                episode_rewards += reward
                f = np.array(pyenv.render(mode="rgb_array")) ./255f0
                push!(frames, deepcopy(f))
                #episode % infotime == 0 && render && Gym.render(env)
                done && break # this breaks it after every episode!
            end
            anim = @animate for bib in frames
                Plots.plot(colorview(RGB, permutedims(bib, [3,1,2])), xticks=[],yticks=[])
            end
            gif(anim, "lunar_lander_anim_2_$num_anim.gif", fps=30)
            num_anim +=1
        end
        push!(histories, history)
        avgreward = 0.1 * episode_rewards + avgreward * 0.9
        if episode % infotime == 0
            println("(episode:$episode, avgreward:$avgreward)")
            close(env)
        end
        if episode % 5 == 0
            Flux.train!(mean_mean_ac_loss, Flux.params(valuenet, policynet), [[histories,policynet, valuenet]], opt_p)
            histories = []
        end
        if episode % 50 == 0
            deep_value_net = deepcopy(valuenet)
        end
        #Flux.train!(mean_ac_loss, Flux.params(valuenet, policynet), [[history]], opt_p)
        vloss = replay(memory,opt_v, valuenet, deep_value_net)
        #println("tloss: $tloss")
        push!(ep_rewards, episode_rewards)
        push!(plosses, mean_ac_loss(history, policynet, valuenet).data)
        push!(vlosses, vloss.data)
        if num_anim >=20
            break
        end
    end
    return ep_rewards, plosses, vlosses
end
main()
