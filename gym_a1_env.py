from lib.env.a1 import A1Robot, A1Params
import time
import numpy as np
from PIL import Image

params = A1Params()
params.if_render = True
robot = A1Robot(params)

i = 1
while 1:
    i += 1
    with open("lib/env/a1/mocap.txt", "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            frame = currentline[0]
            t = currentline[1]
            joints = currentline[2:14]
            joints = np.array(joints, dtype=float)
            robot.step(joints)
            time.sleep(1. / 500.)

    image = robot.render("rgb_array")
    # im = Image.fromarray(image)
    # im.save("test_cv.jpeg")

    print(image)

    if i % 5 == 0:
        robot.reset()


