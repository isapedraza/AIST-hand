import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

# Example coordinates of hand joints (x, y, z) for a single hand
joints0 = np.array([
    [0.537075817584991, 0.579371094703674, 3.91933753007834E-07],  # Wrist
    [0.515692055225372, 0.616869330406189, -0.0277371313422918],  # Thumb_CMC
    [0.512830853462219, 0.658717036247253, -0.0408027246594429],  # Thumb_MCP
    [0.511546492576599, 0.695987701416016, -0.0478233583271503],  # Thumb_IP
    [0.513696849346161, 0.726681232452393, -0.0532500334084034],  # Thumb_TIP
    [0.568946540355682, 0.652715384960175, -0.03798783197999],  # Index_MCP
    [0.596281409263611, 0.702528059482574, -0.0469274371862412],  # Index_PIP
    [0.607759118080139, 0.734170913696289, -0.0497478023171425],  # Index_DIP
    [0.611897170543671, 0.757252216339111, -0.0496565885841846],  # Index_TIP
    [0.588025808334351, 0.636388123035431, -0.024784903973341],  # Middle_MCP
    [0.618966221809387, 0.683611631393433, -0.0303007382899523],  # Middle_PIP
    [0.62541139125824, 0.713803052902222, -0.0292431749403477],  # Middle_DIP
    [0.620976090431213, 0.73240864276886, -0.0271605998277664],  # Middle_TIP
    [0.596882462501526, 0.625758707523346, -0.0126515096053481],  #Ring_MCP
    [0.620478808879852, 0.670611917972565, -0.0169409420341253],  #Ring_PIP
    [0.624352991580963, 0.697729587554932, -0.0174736771732569],  #Ring_DIP
    [0.621739447116852, 0.715920150279999, -0.0165975131094456],  #Ring_TIP
    [0.599073171615601, 0.622693181037903, -0.00243008206598461],  #Pinky_MCP 
    [0.616257131099701, 0.6606525182724, -0.00531668541952968],  #Pinky_PIP
    [0.619439244270325, 0.683303773403168, -0.00530795333907008],  #Pinky_DIP
    [0.617703795433044, 0.6995729804039, -0.00409400835633278]   #Pinky_TIP 
])

joints3 = np.array([
    [0.4678725600242615, 0.7685302495956421, 1.4630057876274805e-07],  # Wrist
    [0.5104346871376038, 0.6653850078582764, -0.015729304403066635],  # Thumb_CMC
    [0.577692985534668, 0.6014739871025085, -0.027911389246582985],  # Thumb_MCP
    [0.6402965188026428, 0.5775749683380127, -0.03234918415546417],  # Thumb_IP
    [0.6788432598114014, 0.5819369554519653, -0.03513678163290024],  # Thumb_TIP
    [0.6181421279907227, 0.7058109045028687, -0.06281488388776779],  # Index_MCP
    [0.71756511926651, 0.7089056372642517, -0.07482714205980301],  # Index_PIP
    [0.7670393586158752, 0.701642632484436, -0.0733497366309166],  # Index_DIP
    [0.796802282333374, 0.6898584365844727, -0.06876683235168457],  # Index_TIP
    [0.6165797710418701, 0.7794047594070435, -0.055213674902915955],  # Middle_MCP
    [0.7195748090744019, 0.7733756303787231, -0.06413058191537857],  # Middle_PIP
    [0.7620394825935364, 0.7491633296012878, -0.05328426882624626],  # Middle_DIP
    [0.7794041633605957, 0.7237311005592346, -0.042667847126722336],  # Middle_TIP
    [0.6087590456008911, 0.8345614075660706, -0.044556889683008194],  #Ring_MCP
    [0.7005959749221802, 0.8148327469825745, -0.04752618446946144],  #Ring_PIP
    [0.7334763407707214, 0.7830122709274292, -0.03276284039020538],  #Ring_DIP
    [0.7439666390419006, 0.75467449426651, -0.02017378807067871],  #Ring_TIP
    [0.5968167781829834, 0.8724871873855591, -0.034181494265794754],  #Pinky_MCP 
    [0.6612139344215393, 0.8397679328918457, -0.03434901684522629],  #Pinky_PIP
    [0.684951663017273, 0.8069558143615723, -0.023003162816166878],  #Pinky_DIP
    [0.6938918232917786, 0.7815890908241272, -0.011583917774260044]   #Pinky_TIP 
])


joints = np.array([
    [0.3781017065048218, 0.7683463096618652, 8.781288585169023e-08],  # Wrist
    [0.3957866430282593, 0.7106790542602539, -0.003160198451951146],  # Thumb_CMC
    [0.42347583174705505, 0.6747949719429016, -0.007918299175798893],  # Thumb_MCP
    [0.44164136052131653, 0.6523035764694214, -0.010374592617154121],  # Thumb_IP
    [0.4503270089626312, 0.6415202021598816, -0.01206372119486332],  # Thumb_TIP
    [0.4579850137233734, 0.7268877625465393, -0.02644542045891285],  # Index_MCP
    [0.5085925459861755, 0.7421081066131592, -0.03631901368498802],  # Index_PIP
    [0.5312881469726562, 0.7324314117431641, -0.036949459463357925],  # Index_DIP
    [0.5390834212303162, 0.7168968319892883, -0.03315439820289612],  # Index_TIP
    [0.4569171071052551, 0.7613038420677185, -0.025968939065933228],  # Middle_MCP
    [0.5145022869110107, 0.7694839239120483, -0.03508846461772919],  # Middle_PIP
    [0.5370080471038818, 0.7525614500045776, -0.02968481183052063],  # Middle_DIP
    [0.5434051752090454, 0.7347351312637329, -0.021539436653256416],  # Middle_TIP
    [0.4528070390224457, 0.7882863879203796, -0.02449456974864006],  #Ring_MCP
    [0.5054843425750732, 0.7935448884963989, -0.029157869517803192],  #Ring_PIP
    [0.5279009342193604, 0.778974711894989, -0.02267627604305744],  #Ring_DIP
    [0.5368617177009583, 0.764393150806427, -0.014417070895433426],  #Ring_TIP
    [0.4463684558868408, 0.8070904016494751, -0.022796157747507095],  #Pinky_MCP 
    [0.48808804154396057, 0.8098113536834717, -0.024927029386162758],  #Pinky_PIP
    [0.5099388360977173, 0.8006063103675842, -0.020404279232025146],  #Pinky_DIP
    [0.522819459438324, 0.7891895174980164, -0.013721258379518986]   #Pinky_TIP 
])


# Assuming 'joints' is your numpy array with shape (n_joints, 3) where columns are x, y, z
mean = joints.mean(axis=0)  # Calculate the mean for x, y, z
std = joints.std(axis=0)    # Calculate the standard deviation for x, y, z

normalized_joints = (joints - mean) / std


# Manual connections
edge_origins = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 5, 9, 10, 10, 11, 11, 12, 13, 9, 13, 14, 14, 15, 15, 16, 17, 13, 17, 18, 17, 18, 19, 19, 20]
edge_ends = [1, 5, 17, 0, 2, 1, 3, 2, 4, 3, 0, 6, 5, 7, 6, 8, 7, 5, 9, 10, 9, 11, 10, 12, 11, 9, 13, 14, 13, 15, 14, 16, 15, 13, 17, 18, 17, 0, 19, 18, 20, 19]

#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

# Calculate distances
distance_wrist_thumb = np.linalg.norm(joints[0] - joints[4])
distance_wrist_index = np.linalg.norm(joints[0] - joints[8])

print("Distance between Wrist and Thumb_TIP (Original):", distance_wrist_thumb)
print("Distance between Wrist and Index_TIP (Original):", distance_wrist_index)



# Calculate distances for normalized joints
distance_wrist_thumb_normalized = np.linalg.norm(normalized_joints[0] - normalized_joints[4])
distance_wrist_index_normalized = np.linalg.norm(normalized_joints[0] - normalized_joints[8])

print("Distance between Wrist and Thumb_TIP (normalized):", distance_wrist_thumb_normalized)
print("Distance between Wrist and Index_TIP (normalized):", distance_wrist_index_normalized)

fig = plt.figure(figsize=(12, 6))

# Original Coordinates
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='blue', label='Original')
ax1.set_title("Original Coordinates")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Scaled Coordinates
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(normalized_joints[:, 0], normalized_joints[:, 1], normalized_joints[:, 2], color='red', label='Normalized')
ax2.set_title("Normalized Coordinates")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()


# Add edges
for start, end in zip(edge_origins, edge_ends):
    x = [joints[start, 0], joints[end, 0]]
    y = [joints[start, 1], joints[end, 1]]
    z = [joints[start, 2], joints[end, 2]]
    ax1.plot(x, y, z, 'gray')  # Draw lines in gray to connect the nodes
    
for start, end in zip(edge_origins, edge_ends):
    x = [normalized_joints[start, 0], normalized_joints[end, 0]]
    y = [normalized_joints[start, 1], normalized_joints[end, 1]]
    z = [normalized_joints[start, 2], normalized_joints[end, 2]]
    ax2.plot(x, y, z, 'gray')  # Draw lines in gray to connect the nodes    

# Plotting the original joints
#ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='blue', label='Original')
# Plotting the normalized joints
#ax.scatter(normalized_joints[:, 0], normalized_joints[:, 1], normalized_joints[:, 2], color='red', label='Normalized')

#ax.set_xlabel('X Coordinate')
#ax.set_ylabel('Y Coordinate')
#ax.set_zlabel('Z Coordinate')
#ax.legend()
#ax.set_title('3D Visualization of Hand Joints with Connections')

plt.show()

