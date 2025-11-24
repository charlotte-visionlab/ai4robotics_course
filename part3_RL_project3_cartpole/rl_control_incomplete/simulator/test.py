from cartpole import CartPoleEnv

if __name__ == '__main__':
    env = CartPoleEnv()
    env.reset()
    env.render()
    input("")

    for i in range(10):
        env.reset()
        done = False
        while not done:
            env.render()
            obs, rew, done, info = env.step(env.action_space.sample())
            import time
            time.sleep(0.025)

        time.sleep(0.5)
