using LightGraphs
using Plots
using DataFrames
using ProgressMeter
using Statistics
using Stheno
using Optim

function sim(n, deg, len, pcontact, pdeath)
    p = deg / n
    g = erdos_renyi(n, p)
    suspectible = Set(1:n)
    agent0 = rand(1:n)
    pop!(suspectible, agent0)
    infected = Dict(agent0=>1)
    state = DataFrame(s=n-1, i=1, r=0, d=0)

    # we need this to synchronize actions
    to_infect = Int[]

    tick = 1
    while length(infected) > 0
        empty!(to_infect)
        cur_dead = 0
        cur_recovered = 0
        for a in suspectible
            nei = neighbors(g, a)
            for b in neighbors(g, a)
                if rand() < pcontact && b in keys(infected)
                    push!(to_infect, a)
                end
            end
        end
        for b in keys(infected)
            nei = neighbors(g, b)
            for a in neighbors(g, b)
                if rand() < pcontact && a in keys(infected)
                    push!(to_infect, a)
                end
            end
            if infected[b] + len > tick
                if rand() < pdeath
                    cur_dead += 1
                    if b in suspectible
                        pop!(suspectible, b)
                        pop!(infected, b)
                    end
                else
                    cur_recovered += 1
                    pop!(infected, b)
                end
                # in Julia it is safe to remove keys from Dict when iterating
                # but not to add keys
                
            end
        end
        println(cur_dead)
        tick += 1
        for a in to_infect
            if a in suspectible # to_infect may contain duplicates
                #pop!(suspectible, a)
                infected[a] = tick
            end
        end
        r = state.r[end] + cur_recovered
        d = state.d[end] + cur_dead
        push!(state, (length(suspectible), length(infected), r, d))
        #println(state)
    end
    return state
end

@time res_1M = sim(1_000_000, 10, 14, 0.5, 0.04)
extrema(sum.(eachrow(res_1M)))
plot(Matrix(res_1M), label=["S" "I" "R" "D"])

df = DataFrame()
@showprogress "Simulating ..." for pcontact in 0.0:0.05:1, _ in 1:64
    println(pcontact)
    push!(df, (pcontact=pcontact, death = sim(10_000, 10, 5, pcontact, 0.04).d[end] / 10_000))
end
println(df)

df_agg = by(df, :pcontact) do x
    (mean=mean(x.death), varmean=var(x.death)/length(x.death))
end



function unpack(θ)
    σ² = exp(θ[1]) + 1e-6
    l = exp(θ[2]) + 1e-6
    return σ², l
end

function npml(θ)
    σ², l = unpack(θ)
    k = σ² * stretch(Matern52(), 1 / l)
    f = GP(k, GPC())
    return -logpdf(f(df_agg.pcontact, df_agg.varmean), df_agg.mean)
end

results = Optim.optimize(npml, randn(2), NelderMead())
σ²_ml, l_ml = unpack(results.minimizer)

k = σ²_ml * stretch(Matern52(), 1 / l_ml);
f = GP(k, GPC());
f_posterior_ml = f | Obs(f(df_agg.pcontact, df_agg.varmean), df_agg.mean);

scatter(df_agg.pcontact, df_agg.mean, color=:red, label="")
display(plot!(f_posterior_ml(0.0:0.001:1.0); label="", color=:blue))
savefig("/Users/kuba/Documents/Side-projects/gym_retro/homeworks/more_ill2.png")
