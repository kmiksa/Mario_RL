println("work")
s = Set(1:4)
println(s)
agent0 = rand(1:2)
println(agent0)
#println(pop!(s, agent0))
pop!(s, agent0)
println(s)

using DataFrames

state = DataFrame(s=3-1, i=1, r=0, d=0)
println(state)

df = DataFrame()
for pcontact in 0.0:0.05:1.0, _ in 1:2
    push!(df, (pcontact=pcontact, s = 2, i =1 , r=1, d= 1 ))
end

println(df)

#df_agg = by(df, :pcontact) do x
#    println(x)
#end
