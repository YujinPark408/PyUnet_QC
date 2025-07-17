import h5py
import glob
import os

# 패턴에 맞는 모든 .h5 파일을 순회
for src_path in sorted(glob.glob("g4-rec-*.h5")):
    # 예: src_path = "g4-rec-3.h5"
    base, ext = os.path.splitext(src_path)
    dst_path = f"{base}_zero{ext}"  # 예: "g4-rec-3_zero.h5"
    print(f"Processing {src_path} → {dst_path}")

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # 숫자형 그룹 이름만 골라, 정렬
        groups = sorted(
            [g for g in src.keys() if g.isdigit()],
            key=lambda x: int(x)
        )
        # 시작 인덱스 (예: 0, 1, 3, 4, 5 중 첫 번째)
        start = int(groups[0])

        # 각 그룹을 'old–start' 만큼 이름 재부여해 복사
        for old in groups:
            new = str(int(old) - start)
            src.copy(old, dst, name=new)

    print(f"  → Done. Groups renamed from {groups[0]}–{groups[-1]} to 0–{len(groups)-1}")

