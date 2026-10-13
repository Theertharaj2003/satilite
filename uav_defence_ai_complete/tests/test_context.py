from src.context.trajectory import TrajectoryStore


def test_trajectory_store():
    store = TrajectoryStore(5)
    state = store.update(1, (0, 0, 10, 10), fps=10)
    assert state.track_id == 1
    store.update(1, (10, 0, 20, 10), fps=10)
    state2 = store.update(1, (20, 0, 30, 10), fps=10)
    assert state2.speed > 0
