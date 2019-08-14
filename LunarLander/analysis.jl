using BSON
using Statistics
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


polrewards, polplosses, polvlosses = BSON.load("lunar_lander_standard_policy_gradients.bson")[:a]
AIrewards, AIplosses, AIvlosses = BSON.load("lunar_lander_standard_active_inference.bson")[:a]
ACrewards, ACplosses, ACvlosses = BSON.load("lunar_lander_actor_critic_2.bson")[:a]
Qrewards, Qvlosses = BSON.load("lunar_lander_q_learning_baseline.bson")[:a]
PGrewards, PGplosses, PGvlosses = BSON.load("lunar_lander_standard_policy_gradients_2.bson")[:a]
AIQrewards, AIQplosses, AIQvlosses = BSON.load("lunar_lander_AI_Q_entropy.bson")[:a]
AC2rewards, AC2plosses, AC2vlosses = BSON.load("lunar_lander_actor_critic.bson")[:a]

function ablation_graph()
    plt = plot(get_mean(entrewards),
     label="Active-Inference-No-Tmodel",
        xlabel="Episode",
        ylabel="Mean Reward",
        title="Ablation Study of Active Inference Algorithm",
        legend=:bottomright,
        lw=0.1,
        size=(800,500),
        framestyle=:box,
        margin = 5mm,
        top_margin=1mm,
        dpi=500)
    plot!(get_mean(AIrewards), label="Active-Inference-No-Entropy",lw=0.1)
    plot!(get_mean(Trewards), label="Active Inference", lw=0.1)
    return plt
end

function lunar_lander_comparison_graph()
    plt = plot(get_mean(polrewards), label="Policy Gradient",
        xlabel ="Episode",
        ylabel = "Mean Reward",
        title="Lunar Lander Algorithm Comparisons",
        legend=:bottomright,
        lw = 0.1,
        size=(800,500),
        framestyle=:box,
        margin=5mm,
        top_margin=1mm,
        dpi=500)
    plot!(get_mean(AIQrewards), label="Active Inference", lw=0.1, color=:red)
    plot!(get_mean(Qrewards), label="Q-learning", lw=0.1,color=:green)
    plot!(get_mean(ACrewards), label="Actor-Critic",lw=0.1,color=:brown)
    #plot!(get_mean(AIrewards), label="AI",lw=0.1)
    return plt
end
plt = lunar_lander_comparison_graph()
savefig("Lunar_Lander_Comparison_Graph.png")


plot(get_mean(polrewards))
plot!(get_mean(AIrewards))
plot!(get_mean(ACrewards))
plot!(get_mean(Qrewards))
plot!(get_mean(PGrewards))
plot!(get_mean(AIQrewards))
plot!(get_mean(AC2rewards))

plot(get_mean(polplosses))
plot!(get_mean(AIplosses))
plot!(get_mean(ACplosses))
plot!(get_mean(PGplosses))
plot!(get_mean(AIQplosses))
plot!(get_mean(AC2plosses))

plot(get_mean(polvlosses))
plot!(get_mean(AIvlosses))
plot!(get_mean(ACvlosses))
plot!(get_mean(Qvlosses))
plot!(get_mean(PGvlosses))
plot!(get_mean(AIQvlosses))
plot!(get_mean(AC2vlosses))


Qrewards[1]
Qrewards[2]

get_mean(Qrewards)

plot(get_mean(polrewards))
plot!(get_mean(AIrewards))
plot!(get_mean(ACrewards))
plot!(get_mean(AIQrewards))
plot!(get_mean(Qrewards))

plot(get_mean(entplosses))
plot!(get_mean(polplosses))
plot!(get_mean(AIplosses))

plot(get_mean(entvlosses))
plot!(get_mean(polvlosses))
plot!(get_mean(AIvlosses))

plot(entrewards)
plot(polrewards)
plot(AIrewards)
