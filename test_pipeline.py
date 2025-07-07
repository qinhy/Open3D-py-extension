import time
import open3dpypro as pro3d


def measure_fps(gen, test_duration=15, func = lambda imgs:None,
                title='#### profile ####'):
    print(title)
    prev_time = time.time()
    start_time = prev_time
    frame_count = 0
    for imgs in gen:
        func(imgs)

        # Compute elapsed time for one frame
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        prev_time = curr_time  # Update for next iteration
        frame_count += 1

        if elapsed_time > 0:
            print(f"Current FPS = {(1.0 / elapsed_time):.2f} FPS         ", end='\r')

        # Check if test duration has passed
        if curr_time - start_time >= test_duration: break

    # Compute average FPS
    print(f"Test completed: Average FPS = {(frame_count/(time.time()-start_time)):.2f}")


def test1():
    gen = pro3d.generator.NumpyRawFrameFileGenerator(
                            sources=['./data/bunny.npy'],
                            shape_types=[pro3d.ShapeType.XYZ])
    print(gen)
    pipes = [      
        pro3d.processors.Processors.RandomSample(),
        pro3d.processors.Processors.CPUNormals(),
        pro3d.processors.Processors.ZDepthViewer(),
    ]
    pro3d.processors.PointCloudMatProcessors.validate_once(gen,pipes)
    while len(pipes)>0:
        measure_fps(gen, func=lambda imgs:pro3d.processors.PointCloudMatProcessors.run_once(
                imgs,{},pipes),
                title=f"#### profile ####\n{'  ->  '.join([f.title for f in pipes])}")
        pipes.pop()



if __name__ == "__main__":
    test1()
