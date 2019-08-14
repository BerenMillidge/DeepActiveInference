using BSON
using Statistics
using StatsBase
using Plots
using Plots.PlotMeasures

function get_mean(x,f=mean)
    arr = zeros(15000,length(x))
    for i in 1:length(x)
        for j in 1:length(x[1])
            arr[j,i] = x[i][j]
        end
    end
    return f(arr, dims=2)
end

pwd()
entrewards, entplosses, entvlossses = BSON.load("acrobot_active_inference_entropy.bson")[:a]
ACposrewards, ACposplosses, ACposvlosses = BSON.load("acrobot_actor_critic_positive.bson")[:a]
ACrewards, ACplosses, ACvlosses = BSON.load("acrobot_actor_critic.bson")[:a]
AIQrewards, AIQplosses, AIQvlosses = BSON.load("acrobot_AI_Q_entropy_2.bson")[:a]
AIQno_entrewards, AIQno_entplosses, AIQno_entvlosses = BSON.load("acrobot_AI_Q_without_entropy.bson")[:a]
PGrewards, PGplosses, PGvlosses = BSON.load("acrobot_standard_policy_gradients.bson")[:a]
Qrewards, Qvlosses = BSON.load("acroobt_q_learning_baseline.bson")[:a]
AIrewards, AIplosses, AIvlosses = BSON.load("standard_active_inference.bson")[:a]

function acrobot_comparison_graph()
    plt = plot(get_mean(PGrewards), label="Policy Gradient",
        xlabel ="Episode",
        ylabel = "Mean Reward",
        title="Acrobot Algorithm Comparisons",
        legend=:bottomright,
        lw = 0.1,
        size=(800,500),
        framestyle=:box,
        margin=5mm,
        top_margin=1mm,
        xticks=[0,5000,10000],
        dpi=500,color=:red)
    #plot!(get_mean(AIQrewards), label="Active Inference", lw=0.1, color=:red)
    plot!(get_mean(Qrewards), label="Q-learning", lw=0.1,color=:green)
    #plot!(get_mean(entrewards), label="Active Inference",lw=0.1, color=:red)
    plot!(get_mean(AIrewards), label="Active Inference",color=:blue)
    #plot!(get_mean(AIrewards), label="AI",lw=0.1)
    return plt
end
plt = acrobot_comparison_graph()
savefig("Acrobot_Comparison_Graph.png")

plot(ACrewards)

plot(get_mean(entrewards))
plot!(get_mean(ACposrewards))
plot!(get_mean(ACrewards))
plot!(get_mean(AIQrewards))
plot!(get_mean(AIQno_entrewards))
plot!(get_mean(PGrewards))
plot!(get_mean(Qrewards))
plot!(get_mean(AIrewards))

plot(get_mean(entplosses))
