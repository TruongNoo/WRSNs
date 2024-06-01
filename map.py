import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Đọc tệp YAML
file_path = r'D:\\Code\\Python\BaoCaoNghienCuu\WRSN\\physical_env\\network\\network_scenarios\\hanoi1000n50.yaml'
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

# Lấy tọa độ của nodes và targets
nodes = data['nodes']
targets = data['targets']
base_station = data['base_station']

# Tạo danh sách tọa độ x và y của nodes, targets và base station
node_x = [node[0] for node in nodes]
node_y = [node[1] for node in nodes]
target_x = [target[0] for target in targets]
target_y = [target[1] for target in targets]
bs_x, bs_y = base_station

# Load hình nền là bản đồ Hà Nội
map_img = mpimg.imread(r'D:\\Code\\Python\BaoCaoNghienCuu\WRSN\\images\\map.png')

# Vẽ bản đồ Hà Nội
plt.figure(figsize=(10, 10))
plt.imshow(map_img)

# Đánh dấu nodes và targets trực tiếp lên ảnh map_img
plt.scatter(node_x, node_y, color='red', label='Nodes')
plt.scatter(target_x, target_y, color='blue', label='Targets')  # Thêm đánh dấu cho targets nếu cần

for node in nodes:
    circle = Circle((node[0], node[1]), radius=80.1, color='LimeGreen', alpha=0.3)
    plt.gca().add_patch(circle)

# Đặt biểu tượng của base station vào hình ảnh
icon_bs = mpimg.imread(r'D:\\Code\\Python\BaoCaoNghienCuu\\Test\\images\bs.png')
imagebox = OffsetImage(icon_bs, zoom=0.05)
ab = AnnotationBbox(imagebox, (bs_x, bs_y), frameon=False)
plt.gca().add_artist(ab)

plt.axis('off')
plt.title('Map of Hanoi with Nodes and Targets')
plt.legend()
plt.show()
