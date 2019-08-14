
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

function replay(opt_v)
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
#basenet = Dense(4,100, Flux.relu)
valuenet = Chain(Dense(4,100, Flux.relu),Dense(100,2))
policynet = Chain(Dense(4,100, Flux.relu), Dense(100,2))
deep_value_net = deepcopy(valuenet)

function sample_action(probs)
    @assert size(probs, 2) == 1
    cprobs = cumsum(probs, dims=1)
    sampled = cprobs .> rand()
    return mapslices(argmax, sampled, dims=1)[1] 
end

function loss(history)
    nS, nA = history.nS, history.nA
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    p = policynet(states)
    V = valuenet(states)

    inds = history.actions + nA*(0:M-1)
    lp = logsoftmax(p)[inds]
    V = V[inds]
    ploss = mean(lp .* V.data)
    return ploss #+ vloss #+ entloss
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

mean_mean_ac_loss(histories) = mean(mean_ac_loss.(histories))

using PyCall
using Plots
using Images, ImageView

np = pyimport("numpy")
L2Reg(x) = mean(x .* x)
function main(
    hidden = [100], # width inner layers
    lr = 1e-2,
    γ = 0.99, #discount rate
    episodes = 10000,
    render = true,
    infotime = 50)
    env = GymEnv("CartPole-v1")
    seed = -1
    seed > 0 && (Random.seed!(seed); Gym.seed!(env, seed))

    global deep_value_net
    opt_p=ADAM(0.001)
    opt_v = ADAM(0.001)
    nS, nA = STATE_SIZE, ACTION_SIZE
    avgreward = 0
    histories = []
    ep_rewards = []
    vlosses = []
    plosses = []
    tlosses = []
    num_anim = 1
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
        state = reset!(env)
        #episode_rewards = 0
        #history = History(nS, nA, γ)
        if avgreward > 450 && num_anim < 20
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
            println("frames: $(length(frames)), $(typeof(frames))")
            anim = @animate for bib in frames
                Plots.plot(colorview(RGB, permutedims(bib, [3,1,2])),xticks=[], yticks=[])
            end
            gif(anim, "cartpole_anim_$num_anim.gif", fps=30)
            num_anim +=1
        end
        push!(histories, history)
        avgreward = 0.1 * episode_rewards + avgreward * 0.9
        if episode % infotime == 0
            println("(episode:$episode, avgreward:$avgreward)")
            close(env)
        end
        if episode % 5 == 0
            Flux.train!(mean_mean_ac_loss, Flux.params(valuenet, policynet), [[histories]], opt_p)
            histories = []
        end
        if episode % 50 == 0
            deep_value_net = deepcopy(valuenet)
        end
        #Flux.train!(mean_ac_loss, Flux.params(valuenet, policynet), [[history]], opt_p)
        vloss = replay(opt_v)
        #println("tloss: $tloss")
        push!(ep_rewards, episode_rewards)
        push!(plosses, mean_ac_loss(history))
        push!(vlosses, vloss)
        if num_anim >= 20
            break
        end
    end
    return ep_rewards, plosses, vlosses
end
main()

