import Utils as utils
from ConfigLoader import logging
from VFCWorks import VFCWorks

def first_exec(main_work, file_path, video_out_path, face_out_path):
    # 1. video -> img
    print(f"Main_debug -> Process File: Demo to Image")
    main_work.process_video_export(file_path, video_out_path)

    # 2. img -> [face]
    print(f"Main_debug -> Process File: Demo FACE DETECT")
    main_work.process_face_export(video_out_path)

    # 2.1 首次执行的任务可在这里导出数据，后继仅尝试不同相似度阈值聚类时，此处可不用重复导出
    print(f"Main_debug -> Export data")
    main_work.export_data()

    # 3. [face] -> similarity
    print(f"Main_debug -> Process File: Demo FACE SIMILARITY")
    main_work.process_face_similarity()

    # 4. [face] -> img
    print(f"Main_debug -> Process File: Demo FACE EXPORT")
    main_work.export_faces_cluster(face_out_path)

    # 5. end
    main_work.show_cost()
    main_work.show_debug_values()
    print(f"Main_debug -> Process File: Demo Finished")


def retry_exec(main_work, face_out_path):
    # 1. clear face_out_path
    if utils.os.path.exists(face_out_path):
        utils.shutil.rmtree(face_out_path)
    utils.os.makedirs(face_out_path, exist_ok=True)

    # 2.2 import data (last task)
    # 调整相似度阈值后，不需要再从头执行，从此处开始执行（屏蔽前置步骤），导入已有数据，直接聚类，首次执行的任务需屏蔽此处
    print(f"Main_debug -> Import data")
    main_work.import_data()

    # 3. [face] -> similarity
    print(f"Main_debug -> Process File: Demo FACE SIMILARITY")
    main_work.process_face_similarity()

    # 4. [face] -> img
    print(f"Main_debug -> Process File: Demo FACE EXPORT")
    main_work.export_faces_cluster(face_out_path)

    # 5. end
    main_work.show_cost()
    main_work.show_debug_values()
    print(f"Main_debug -> Process File: Demo Finished")


def func_call():
    main_work = VFCWorks()
    # [(file_idx_name, filename, file_path, video_out_path, face_out_path)]
    file_path=("./workspace/demo3_720p3h.mkv")
    video_out_path="./workspace/Demo"
    face_out_path="./workspace/Demo/faceset"

    # 手动二选一切换执行
    first_exec(main_work, file_path, video_out_path, face_out_path)
    #retry_exec(main_work, face_out_path)

if __name__ == '__main__':
    print("Start...")
    logging.getLogger().setLevel(logging.INFO)
    func_call()
    print("End...")
    exit(0)
