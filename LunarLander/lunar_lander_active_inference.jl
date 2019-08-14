using OpenAIGym
import Random
using Statistics
using Flux
using StatsBase
using Plots
using ImageMagick
using Images
using ImageView
const MEM_SIZE = 100000
const BATCH_SIZE = 200
const STATE_SIZE = 8
const ACTION_SIZE =4

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

function replay(opt_v, valuenet, deep_value_net)
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
    #println("ploss: $ploss")
    return ploss
end

mean_mean_ac_loss(histories,policynet, valuenet) = mean([mean_ac_loss(hist, policynet, valuenet) for hist in histories])

function main(
    γ = 0.99, #discount rate
    episodes = 15000,
    infotime = 50,
    make_anim = true)
    env = GymEnv("LunarLander-v2")
    #seed = -1
    #eed > 0 && (Random.seed!(seed); Gym.seed!(env, seed))

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
            remember(state, action, reward, next_state, done)
            state = next_state
            episode_rewards += reward

            #episode % infotime == 0 && render && Gym.render(env)
            done && break # this breaks it after every episode!
        end

        push!(histories, history)
        avgreward = 0.1 * episode_rewards + avgreward * 0.9
        if episode % infotime == 0
            if make_anim
                println("preparing animation!")
                state = reset!(env)
                frames = []
                for t=1:10000
                    println("$t")
                    p = policynet(state)
                    p = softmax(p)
                    action = sample_action(p.data)

                    reward, next_state = step!(env, action-1)
                    append!(history.states, state)
                    push!(history.actions, action)
                    push!(history.rewards, reward)
                    done = env.done
                    remember(state, action, reward, next_state, done)
                    state = next_state
                    episode_rewards += reward
                    f = render(env, mode=:rgb_array)
                    arr = zeros(400,600,3)
                    for i in 1:length(f)
                        for j in 1:length(get(f, i-1))
                            for k in 1:length(get(f, (i-1,j-1)))
                                arr[i,j,k] = get(f, (i-1,j-1,k-1))
                            end
                        end
                    end
                    push!(frames, arr)

                    #episode % infotime == 0 && render && Gym.render(env)
                    done && break # this breaks it after every episode!
                end
                println(length(frames))
                anim = @animate for i in 1:length(frames)
                    plot(colorview(RGB, permutedims(frames[i], [3,1,2])))
                end
                gif(anim, "Actor_Critic/animations/lunar_lander_ep_$episode.gif", fps = 30)
            end
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
        vloss = replay(opt_v, valuenet, deep_value_net)
        #println("tloss: $tloss")
        push!(ep_rewards, episode_rewards)
        push!(plosses, mean_ac_loss(history,policynet, valuenet).data)
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
      BSON.bson("results/lunar_lander_standard_active_inference.bson", a=[rs,pls,vls])
      println("save successful!")
    end
end

main()
#

#save_results()
using PyCall

env = GymEnv("LunarLander-v2")
s = reset!(env)
using PyCall
f = render(env, mode=:rgb_array)
f.view()
arr = zeros(400,600,3)
for i in 1:length(frame)
    for j in 1:length(get(frame, i-1))
        for k in 1:length(get(frame, (i-1,j-1)))
            arr[i,j,k] = get(frame, (i-1,j-1,k-1))
        end
    end
end
f.tobytes()
@time f.tolist()
function get_arr(f)
    arr = zeros(400,600,3)
    for i in 1:length(f)
        for j in 1:length(get(f, i-1))
            for k in 1:length(get(f, (i-1,j-1)))
                arr[i,j,k] = get(f, (i-1,j-1,k-1))
            end
        end
    end
    return arr
end
@time get_arr(f.tolist())
