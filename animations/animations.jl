using OpenAIGym
import Random
using Statistics
using Flux
using StatsBase
using PyCall
env = GymEnv("CartPole-v1")
s = reset!(env)
a = 1
r,sdash = step!(env,a)
frame = render(env, mode=:rgb_array)
f = convert(Array,frame) 
zs = zeros(255,255,255)
frame[1]


env = GymEnv("LunarLander-v2")
s = reset!(env)
frame = render(env, mode=:rgb_array)
size(frame)
env.pyenv
convert(Array, frame)

using Plots
using Images
using ImageView

pyenv = env.pyenv
gym = pyimport("gym")
np = pyimport("numpy")

f = np.array(pyenv.render(mode="rgb_array")) ./255f0
colorview(RGB, permutedims(f, [3,1,2]))
Plots.plot(f)

py"""
import numpy as np

def get_frame(env):
    f = env.render(mode="rgb_array")
    print(type(f))

"""
Int(rand() > 0.5)
py"get_frame"(pyenv)

env
env = GymEnv("CartPole-v1")
pyenv = env.pyenv
sample(OpenAIGym.actionset(env.pyenv))
Plots.plot([1,2,3])
frames = []
s = reset!(env)
for i in 1:100
    println("$i")
    a = Int(rand() > 0.5)
    r,sdash = step!(env, a)
    f = np.array(pyenv.render(mode="rgb_array")) ./255f0
    push!(frames, convert(Array{Float32},f))
end
frames
frames[1]
Plots.plot(colorview(RGB, permutedims(frames[56], [3,1,2])))
cols = []
for bib in frames
    col = colorview(RGB, permutedims(bib, [3,1,2]))
    push!(cols, col)
end
cols
Plots.plot(cols[1])
frame = 0
@gif for col in cols
    Plots.plot(col)
end

@gif for bib in frames
    Plots.plot(colorview(RGB, permutedims(bib, [3,1,2])))
end

anim = @animate for bib in frames
    Plots.plot(colorview(RGB, permutedims(bib, [3,1,2])))
end
gif(anim, "test_cartpole.gif", fps=30)
pwd()
