import Utils as utils
from ConfigLoader import logging
from VFCWorks import VFCWorks

def full_auto_work():
    main_work = VFCWorks()
    # [(file_idx_name, filename, file_path, video_out_path, face_out_path)]
    task_list = utils.prepare_workspace()
    for file_idx_name, filename, file_path, video_out_path, face_out_path in task_list:
        main_work.clear_works()
        try:
            # 1. video -> img
            print(f"Main -> Process File: {filename} to Image")
            main_work.process_video_export(file_path, video_out_path)

            # 2. img -> [face]
            print(f"Main -> Process File: {filename} FACE DETECT")
            main_work.process_face_export(video_out_path)

            # 3. [face] -> similarity
            print(f"Main -> Process File: {filename} FACE SIMILARITY")
            main_work.process_face_similarity()

            # 4. [face] -> img
            print(f"Main -> Process File: {filename} FACE EXPORT")
            main_work.export_faces_cluster(face_out_path)

            # 5. end
            main_work.show_cost()
            main_work.show_debug_values()
            print(f"Main -> Process File: {filename} Finished")

        except Exception as e:
            logging.critical(f"Main Error: {e}", exc_info=True)

if __name__ == '__main__':
    print("Start Full Auto Work...")
    full_auto_work()
    print("Full Auto Work Finished!")
    exit(0)
