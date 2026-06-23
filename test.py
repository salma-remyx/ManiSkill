import newton
import warp as wp


def main():
    wp.init()
    sim_device = wp.get_device("cpu")
    model_builder = newton.ModelBuilder()
    model_builder.add_shape_plane()
    model = model_builder.finalize(sim_device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=10)
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.close()
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    sim_timestep = 1.0 / 240.0
    sim_time = 0.0
    while True:
        print("Sim time: ", sim_time)
        state_0.clear_forces()
        model.collide(state_0, contacts)
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
