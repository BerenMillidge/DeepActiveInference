using BSON
using Statistics, StatsBase
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

ACrewards, ACplosses,ACvlosses = BSON.load("actor_critic_baseline.bson")[:a]
Qrewards,Qvlosses = BSON.load("q_learning_baseline.bson")[:a]
PGrewards, PGplosses, PGvlosses = BSON.load("standard_policy_advantage.bson")[:a]
AIrewards, AIplosses, AIvlosses = BSON.load("standard_active_inference.bson")[:a]
entrewards, entplosses, entvlosses = BSON.load("active_inference_entropy_loss.bson")[:a]
Trewards, Tplosses, Tvlosses, Ttlosses = BSON.load("active_inference_tmodel.bson")[:a]
prewards, pplosses, pvlosses, precloss = BSON.load("active_inference_learned_precision.bson")[:a]

plot(get_mean(prewards))
plot!(get_mean(AIrewards))
plot(get_mean(precloss))

function basic_comparison_graph()
    plt = plot(get_mean(ACrewards),
     label="Actor-Critic",
        xlabel="Episode",
        ylabel="Mean Reward",
        title="Comparison of Active Inference with Standard Reinforcement Learning Algorithms",
        legend=:bottomright,
        lw=0.1,
        size=(800,500),
        framestyle=:box,
        margin = 5mm,
        top_margin=1mm,
        dpi=500)
    plot!(get_mean(Qrewards), label="Q-learning",lw=0.1)
    plot!(get_mean(Trewards), label="Active Inference", lw=0.1)
    return plt
end
plt = basic_comparison_graph()
savefig("basic_comparison_graph.png")

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

function ablation_comparison_graph()
    plt = plot(get_mean(AIrewards),
     label="Ablated Active Inference",
        xlabel="Episode",
        ylabel="Mean Reward",
        title="Comparison of Ablated Active Inference with Reinforcement Learning Baselines",
        legend=:bottomright,
        lw=0.1,
        size=(800,500),
        framestyle=:box,
        margin = 5mm,
        top_margin=1mm,
        dpi=500)
    plot!(get_mean(ACrewards), label="Actor-Critic",lw=0.1)
    plot!(get_mean(Qrewards), label="Q-learning", lw=0.1)
    return plt
end
plt = ablation_comparison_graph()
savefig("ablation_comparison_graph.png")

function Tmodel_loss_graph()
    plt = plot([get_mean(Ttlosses),get_mean(Ttlosses)[500:end]],label="", xticks = [0,5000,10000],xlabel="Episode", ylabel="Transition-Model Loss", layout=2)
    return plt
end
plt = Tmodel_loss_graph()
savefig("Transition_Model_graph.png")


plot(get_mean(Tplosses))
plot(get_mean(Ttlosses)[500:end])
plot(get_mean(Tplosses) .+ get_mean(Ttlosses))

qmean = get_mean(Qrewards)
vcat(qmean, zeros(5000,1))

function basic_graph(f=mean,LW=0.1,linalpha = 1)
    plt = plot(get_mean(ACrewards,f),
    label="Actor Critic",
    xlabel="Episode",
    ylabel="Mean Reward",
    title="Comparison of Active Inference with Standard Reinforcement Learning Algorithms",
    legend=:bottomright,
    lw=LW,
    size=(800,500),
    framestyle=:box,
    margin=5mm,
    top_margin=4mm,
    dpi=1000,
    linealpha=linalpha,color=:green,linestyle=:solid)
    plot!(get_mean(Qrewards,f), label="Q-learning",lw=LW,linealpha=linalpha,color=:purple,linestyle=:solid)
    #plot!(get_mean(PGrewards,f),label="Policy Gradients",lw=LW)
    plot!(get_mean(AIrewards,f), label="Active Inference",lw=LW,linealpha=linalpha,color=:red,linestyle=:solid)
    plot!(get_mean(entrewards,f),label="Active Inference Entropy",lw=LW,linealpha=linalpha,color=:blue,linestyle=:solid)
    savefig("median_comparison_graph.png")
    return plt
end
plt = basic_graph(median,0.5,1)

plot(AIrewards, legend=:none, size=(800,500),dpi=500,top_margin=1mm,title="Rewards for all runs of Active Inference Agent", framestyle=:box, margin=5mm, linewidth=0.1, linealpha=0.8,xlabel="Episode",ylabel="Mean Reward")
savefig("AI_agent_all_runs")
plot(Qrewards, legend=:none, size=(800,500),dpi=500, top_margin=1mm, title="Rewards for all runs for Q-learning agent", framestyle=:box, margin=5mm, linewidth=0.1, linealpha=0.8,xlabel="Episode",ylabel="Mean Reward")
savefig("Q_agent_all_runs")
plot(entrewards, legend=:none, size=(800,500),dpi=500, framestyle=:box,top_margin=1mm, title="Rewards for all runs of Active Inference Entropy Agent", margin=5mm, linewidth=0.1, linealpha=0.8,xlabel="Episode",ylabel="Mean Reward")
savefig("Entropy_AI_all_runs.png")
plot(ACrewards, legend=:none, size=(800,500),dpi=500, framestyle=:box,top_margin=1mm, title="Rewards for all runs of Actor Critic Agent", margin=5mm, linewidth=0.1, linealpha=0.8,xlabel="Episode",ylabel="Mean Reward")
savefig("Actor-Critic_all_runs.png")
plot(get_mean(entplosses), legend=:none, size=(800,500),dpi=500, framestyle=:box,top_margin=1mm, title="Mean Free-Energy for Active-Inference Agent", margin=5mm, linewidth=0.1, linealpha=0.8,xlabel="Episode",ylabel="Variational Free-Energy")
savefig("AI_VFE_plot.png")

plot(get_mean(AIplosses))
plot(get_mean(Qvlosses))
plot(get_mean(AIvlosses))
plot(get_mean(ACplosses))
plot(get_mean(ACvlosses))
plot(ACvlosses)
plot(entvlosses)
plot(entplosses)

ACQrewards, ACQplosses, ACQvlosses = BSON.load("actor_critic_q_expectation.bson")[:a]
AIQrewards, AIQplosses, AIQvlosses = BSON.load("standard_active_inference_q_expectation.bson")[:a]

plot(get_mean(ACQrewards))
plot!(get_mean(ACrewards))
plot!(get_mean(AIrewards))
plot!(get_mean(AIQrewards))

plot(AIQrewards)
plot(AIrewards)

plot(get_mean(ACQvlosses))
plot!(get_mean(ACvlosses))
plot!(get_mean(AIvlosses))
plot!(get_mean(AIQvlosses))

plot(get_mean(ACplosses))
plot!(get_mean(ACQplosses))
plot!(get_mean(AIplosses))
plot!(get_mean(AIQplosses))
plot(get_mean(AIQplosses))

OPrewards, OPplosses, OPvlosses = BSON.load("off_policy_active_inference.bson")[:a]
plot(get_mean(OPrewards))
plot!(get_mean(AIrewards))
plot!(get_mean(entrewards))

