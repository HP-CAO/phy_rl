from lib.env.a1 import A1Robot, A1Params
import time
import numpy as np
from PIL import Image

params = A1Params()
params.if_render = True
robot = A1Robot(params)

with open("lib/env/a1/mocap.txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(",")
        frame = currentline[0]
        t = currentline[1]
        starting_motion = currentline[2:14]
        break

stable_motion = starting_motion

i = 1

while 1:
    i += 1
    joints = np.array(stable_motion, dtype=float)
    observation, reward, done, _, reward_list = robot.step(joints)

    # print(reward_list)
    # print(reward)
    time.sleep(1. / 500.)

    # image = robot.render("rgb_array")
    # im = Image.fromarray(image)
    # im.save("test_cv.jpeg")

    # if i % 5 == 0:
    #     robot.reset()
