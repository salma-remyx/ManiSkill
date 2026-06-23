import newton
import warp as wp


def main():
    wp.init()
    sim_device = wp.get_device("cpu")
    model_builder = newton.ModelBuilder()
    model_builder.add_shape_plane()
    model_builder.add_urdf(
        source="/home/stao/work/maniskill/ManiSkill/mani_skill/assets/robots/panda/panda_v2.urdf",
        # xform=wp.transform(wp.vec3(0, 0, 0.2), wp.quat(0, 0, 0, 1)),
    )
    # import ipdb; ipdb.set_trace()
    model_builder.joint_target_ke[:9] = [400]*9
    model_builder.joint_target_kd[:9] = [20]*9
    model_builder.joint_effort_limit[:9] = [87]*9
    model_builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2
    gravcomp_attr = model_builder.custom_attributes["mujoco:jnt_actgravcomp"]
    if gravcomp_attr.values is None:
        gravcomp_attr.values = {}
    for dof_idx in range(7):
        gravcomp_attr.values[dof_idx] = True

    gravcomp_body = model_builder.custom_attributes["mujoco:gravcomp"]
    if gravcomp_body.values is None:
        gravcomp_body.values = {}
    for body_idx in range(2, 14):
        gravcomp_body.values[body_idx] = 1.0

    solimp_attr = model_builder.custom_attributes.get("mujoco:geom_solimp")
    priority_attr = model_builder.custom_attributes.get("mujoco:geom_priority")
    if solimp_attr is not None and priority_attr is not None:
        if solimp_attr.values is None:
            solimp_attr.values = {}
        if priority_attr.values is None:
            priority_attr.values = {}
        for s, b in enumerate(model_builder.shape_body):
            if b in (12, 13):
                solimp_attr.values[s] = (0.7, 0.95, 0.0001, 0.5, 2.0)
                priority_attr.values[s] = 1
    model = model_builder.finalize(sim_device)
    newton.eval_fk(model, model.joint_q + 0.1, model.joint_qd, model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    solver = newton.solvers.SolverMuJoCo(model, )
    viewer = newton.viewer.ViewerGL(paused=True)
    viewer.set_model(model)
    # viewer.close()
    # viewer = newton.viewer.ViewerGL()
    # viewer.set_model(model)
    sim_timestep = 1.0 / 240.0
    sim_time = 0.0
    import ipdb; ipdb.set_trace()
    while True:
        if viewer.should_step():
            state_0.clear_forces()
            model.collide(state_0, contacts)
            control.joint_target_q = wp.zeros_like(model.joint_q)
            solver.step(
                state_in=state_0,
                state_out=state_1,
                control=control,
                contacts=contacts,
                dt=sim_timestep,
            )
            state_0, state_1 = state_1, state_0
            sim_time += sim_timestep
        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.end_frame()


if __name__ == "__main__":
    main()
