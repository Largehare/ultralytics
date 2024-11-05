from ultralytics import YOLO
def main():
    model = YOLO('yolo11-ghost.yaml')
    # model = YOLO('yolov8-ghost.yaml')
    _ = model.to('cuda')
    model.reset_weights()
    results = model.train(data='coco8.yaml', epochs=5, batch=2, workers=2)
    # results.val()

if __name__ == "__main__":
    main()