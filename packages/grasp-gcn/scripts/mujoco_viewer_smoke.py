import mujoco
import mujoco.viewer


XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
