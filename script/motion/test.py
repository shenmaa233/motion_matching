# detailed_compare.py
import numpy as np

def detailed_compare(traj1_path, traj2_path):
    d1 = np.load(traj1_path)
    d2 = np.load(traj2_path)
    
    body1 = d1['body_pos_w']
    body2 = d2['body_pos_w']
    
    print("=== 详细比较 ===")
    
    # 检查是否完全相同
    if np.array_equal(body1, body2):
        print("⚠️ 两个轨迹完全相同！")
        return
    
    print("✓ 两个轨迹不同")
    
    # 检查相对运动是否相同（减去第一帧位置）
    body1_rel = body1 - body1[0:1, :, :]
    body2_rel = body2 - body2[0:1, :, :]
    
    if np.allclose(body1_rel, body2_rel, atol=1e-5):
        print("\n⚠️ 关键发现：两个轨迹的相对运动完全相同！")
        print("   只是整体位置偏移不同，所以可视化看起来一样")
        
        # 计算偏移量
        offset = body2[0, 0, :2] - body1[0, 0, :2]
        print(f"   位置偏移: ({offset[0]:.2f}, {offset[1]:.2f})")
    else:
        print("\n✓ 相对运动也不同，应该是不同的动画")
        
        # 计算相对运动的差异
        rel_diff = np.linalg.norm(body1_rel - body2_rel)
        print(f"   相对运动差异: {rel_diff:.3f}")

detailed_compare("1.npz", "2.npz")