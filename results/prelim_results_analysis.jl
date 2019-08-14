using BSON
using Plots
using Statistics
pwd()

qrewards, qvlosses = BSON.load("results/q_learning_baseline.bson")[:a]
AIrewards, AIplosses, AIvlosses = BSON.load("Actor_Critic/results/standard_active_inference.bson")[:a]
polrewards, polplosses, polvlosses = BSON.load("Actor_Critic/results/standard_policy_advantage.bson")[:a]
entrewards, entplosses, entvlosses = BSON.load("Actor_Critic/results/active_inference_entropy_loss.bson")[:a]
MACrewards, MACplosses, MACvlosses = BSON.load("Actor_Critic/results/mean_actor_critic_without_entropy.bson")[:a]
ACrewards, ACplosses, ACvlosses = BSON.load("Actor_Critic/results/actor_critic_baseline.bson")[:a]
trewards, tplosses, tvlosses = BSON.load("Actor_Critic/results/active_inference_tmodel.bson")[:a]
action_specificrewards, action_specificplosses, action_specificvlosses = BSON.load("Actor_Critic/results/action_specific_Tmodel.bson")[:a]
function get_mean(x,f=mean)
    arr = zeros(15000,20)
    for i in 1:length(x)
        for j in 1:length(x[1])
            arr[j,i] = x[i][j]
        end
    end
    return f(arr, dims=2)
end

mean_q = get_mean(qrewards)
mean_AI = get_mean(AIrewards)
mean_pol = get_mean(polrewards)
mean_ent = get_mean(entrewards)
mean_mac = get_mean(MACrewards)
mean_ac = get_mean(ACrewards)
mean_t = get_mean(trewards)
mean_act = get_mean(action_specificrewards)
plot(mean_q)
plot!(mean_AI)
plot!(mean_pol)
plot!(mean_ent)
plot!(mean_mac)
plot!(mean_ac)
plot!(mean_t)
plot!(mean_act)
m = mean(arr,dims=2)
st = std(arr, dims=2)

plot(m.+st)
plot!(m)
plot!(m .-st)
plot(std(arr,dims=2))
plot(maximum(arr,dims=2))
plot(minimum(arr,dims=2))
plot(mean(arr, dims=2))

plot(rewards[14])
function plot_all(x)
    plt = plot(x[1])
    for i in 1:length(x)-1
        plot!(x[i+1])
    end
    return plt
end
plot_all(rewards)
savefig("test_vlosses.png")


AC_rewards, _, _ = BSON.load("Actor_Critic/results/actor_critic_baseline.bson")[:a]
ACQ_rewards, _, _ = BSON.load("Actor_Critic/results/actor_critic_q_expectation.bson")[:a]
AI_rewards, _, _ = BSON.load("Actor_Critic/results/standard_active_inference.bson")[:a]
AIQ_rewards, _, _ = BSON.load("Actor_Critic/results/standard_active_inference_q_expectation.bson")[:a]

mean_AC = get_mean(AC_rewards)
mean_ACQ = get_mean(ACQ_rewards)
mean_AI = get_mean(AI_rewards)
mean_AIQ = get_mean(AIQ_rewards)

plot(mean_AC)
plot!(mean_ACQ)
plot!(mean_AI)
plot!(mean_AIQ)
plot_all(entrewards)
plot_all(AI_rewards)
plot_all(AIQ_rewards)

plot_all(AC_rewards)
plot_all(ACQ_rewards)

qrewards, qvlosses = BSON.load("results/q_learning_baseline.bson")[:a]
AIrewards, AIplosses, AIvlosses = BSON.load("results/standard_active_inference.bson")[:a]
polrewards, polplosses, polvlosses = BSON.load("results/standard_policy_advantage.bson")[:a]
entrewards, entplosses, entvlosses = BSON.load("results/active_inference_entropy_loss.bson")[:a]
MACrewards, MACplosses, MACvlosses = BSON.load("results/mean_actor_critic_without_entropy.bson")[:a]
ACrewards, ACplosses, ACvlosses = BSON.load("results/actor_critic_baseline.bson")[:a]
AIQ, AIQplosses, AIQvlosses = BSON.load("results/standard_active_inference_q_expectation.bson")[:a]
trewards, tplosses, tvlosses, ttlosses = BSON.load("results/active_inference_tmodel.bson")[:a]

plot(get_mean(trewards))
plot!(get_mean(entrewards))
plot(get_mean(entplosses))
plot!(get_mean(tplosses))
plot(get_mean(ttlosses)[10000:end])

function comparison_graph(f=mean)
    plt = plot(get_mean(qrewards,f),label="q_agent",xlabel="Episode", ylabel="Mean Reward",plot_title="Test",xticks=[5000,10000,15000])
    plot!(get_mean(AIrewards,f),label="active_inference")
    plot!(get_mean(polrewards,f),label="policy_gradient")
    plot!(get_mean(entrewards,f),label="active_inference_entropy")
    plot!(get_mean(MACrewards,f),label="mean_actor_critic")
    plot!(get_mean(ACrewards,f),label="actor_critic")
    return plt
end

plot_all(qrewards)
plot_all(AIrewards)
plot_all(polrewards)
plot_all(entrewards)
plot_all(MACrewards)
plot_all(ACrewards)
plot(get_mean(AIQ))
rewards, plosses, vlosses = BSON.load("results/results_lunar_lander/lunar_lander_standard_active_inference.bson")[:a]
plot(get_mean(rewards, mean))
plot(AIQvlosses,legend=:none)
plot(AIplosses,legend=:none)
plot(entplosses, legend=:none)
plot(qvlosses,legend=:none)
plot(MACplosses,legend=:none)


flatten(x) = [x[i][1] for i in 1:length(x)]

a = flatten(mean.(AIrewards,dims=1))

bar(flatten(mean.(AIrewards, dims=1)))
comparison_graph(minimum)
plot(polvlosses)
plot(trewards)
