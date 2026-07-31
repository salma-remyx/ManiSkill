# ManiSkill 4 notes

Goals
- Add newton
- Make it possible to support multiple physics, multiple rendering backends; in a easy to understand, maintainable manner.
- Keep the task API design and lifecycle largely the same (e.g load scene, init episode, reset, reconfigures)

a lot of slightly randomly assorted notes on some sim framework design decisions I made.

## Architecture

### Rewriting

I spent a month trying to adapt ManiSkill3 code to be multi-physics and rendering, and that turned out to be a terrible idea. A lot of MS3 code was never written in a way to be amenable to the ambitions of MS4, and we still have a lot of legacy MS2 code that was even harder to try and translate. But the first failure here was a good exercise in understanding how different simulators (newton, sapien etc.) work and what their APIs/contracts are like. Rewriting a lot of the internal sim code for MS4 has since been incredibly refreshing and fun.

### Notes on being tensor backend agnostic

Every popular sim framework usually has some default "tensor backend", whether it's torch, warp, jax etc. Usually this is fine to assume some backend (e.g. IsaacLab is torch based) and there are not that many drawbacks to just using torch. But as an exercise in my programming/system design/simulation understanding skills, and to cover that small 1% case of "we can't use torch", I'm going to try and make ManiSkill4 be tensor backend agnostic.

How to avoid assuming a tensor backend?

For each simulator (here simulator = physics and or rendering engine), it may come with it's own tensor backend. For newton that's warp, for mjx that's jax. Generally all code for a particular simulator should stick to that particular simulation tensor backend. This seems unavoidable. For example, everything inside mani_skill/sim/newton will be purely written in warp.

For simulators that do not use a popular tensor backend like SAPIEN, currently the plan is to default to warp/numpy. The reasoning here is
- warp has some nice operations designed for spatial computation, which happens a lot in simulation
- unlike torch, warp does not have strict requirements on a cuda dependency.

Now at the end of the day, if you want to create some new simulation task in ManiSkill you do need to store data in _some_ kind of tensor. The goal now
is to move the decision-point of where that tensor backend selection is chosen to the `BaseEnv` layer / task definition layer, which is where most users will interact with ManiSkill through.

To support and show people how ManiSkill can be flexible, example implementations of the PushCube task will be implemented with a torch and warp tensor setup, with the warp one also showcasing the power of warp kernels.
