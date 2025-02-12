import os
import io
from pathlib import Path
import time
import numpy as np
import cv2
from PIL import Image
import base64
import torch
from torch import nn
from torchvision import transforms, models
import pdf2image
import matplotlib.pyplot as plt
import uuid
from pyiqa.api_helpers import create_metric

FILEPATH = "./file_storage"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['card', 'administrative_document', 'face']
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize (h,w)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
threshold_cls = 0.8
threshold_det = 0.95
transform = transforms.Compose([
    transforms.ToTensor(),
])
color_threshold = 15  # threshold of coloured-gray
pixel_ratio_threshold = 0.35  # threshold of ratio of coloured (e.g. 40%)


def is_resize(image, max_resize=1500, min_size=500):
    width, height = image.size

    if min(width, height) < min_size:
        return False, image
    if max(width, height) >= max_resize:
        # compute resize ratio
        ratio = max_resize / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        print(f"Resizing image to: {new_width}x{new_height}")

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return True, image

def rotate(image):
    image_to_show = np.copy(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges_image = cv2.Canny(blurred_image, 70, 150, apertureSize=3)  # L2gradient=True
    # cv2.imshow("edge", edges_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Phát hiện các đoạn thẳng dài và thẳng bằng Hough Line Transform
    # lines = cv2.HoughLines(edges_image, 1, np.pi / 180, threshold=100)
    min_line_length = int(0.5 * min(image.shape[:2]))  # Độ dài tối thiểu của đoạn thẳng
    max_line_gap = 20  # Khoảng cách tối đa giữa các đoạn thẳng để nối chúng lại thành một
    lines = cv2.HoughLinesP(edges_image, 1, np.pi / 180, threshold=80, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        # Lọc và chọn lựa đoạn thẳng phù hợp (đoạn thẳng có độ dài > threshold_length)
        threshold_length = 80
        filtered_lines = []
        # for line in lines:
        #     rho, theta = line[0]
        #     if np.pi / 4 < theta < 3 * np.pi / 4:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1000 * (-b))
        #         y1 = int(y0 + 1000 * a)
        #         x2 = int(x0 - 1000 * (-b))
        #         y2 = int(y0 - 1000 * a)
        #         line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        #         if line_length > threshold_length:
        #             filtered_lines.append(((x1, y1), (x2, y2)))
        #         # Vẽ các đường thẳng
        #         cv2.line(image_to_show, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Chỉ chọn các đoạn thẳng nằm ngang hoặc gần nằm ngang
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if line_length > min_line_length:  # abs(angle) < 45 and
                filtered_lines.append(((x1, y1), (x2, y2)))
                cv2.line(image_to_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # cv2.imshow("draw lines", image_to_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if len(filtered_lines) > 0:
            # Lựa chọn đoạn thẳng dài nhất
            longest_line = max(filtered_lines, key=lambda x: np.linalg.norm(np.array(x[0]) - np.array(x[1])))
            x1, y1 = longest_line[0]
            x2, y2 = longest_line[1]
            # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255, 2))

            # Tính góc xoay của đường thẳng
            rotation_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            print("rotate angle:", rotation_angle)
            if 45 <= rotation_angle <= 90:
                # if (90-abs(rotation_angle)) < abs(rotation_angle):
                rotation_angle = rotation_angle - 180


            # Xoay lại ảnh để biển số xe nằm ngang
            height, width = image.shape[:2]
            # print("height, width:", height, width)
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            # rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

            # Tính toán khung chứa mới để bao toàn bộ ảnh sau khi xoay
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])

            # Chiều rộng và chiều cao mới sau khi xoay ảnh
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            # Cập nhật ma trận chuyển đổi để đặt tâm của ảnh tại khung mới
            rotation_matrix[0, 2] += bound_w / 2 - center[0]
            rotation_matrix[1, 2] += bound_h / 2 - center[1]

            # Áp dụng xoay và mở rộng khung chứa ảnh
            rotated_image = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))

            # print("FOUND LINES.")
            print("rotate angle:", rotation_angle)
        else:
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1), plt.imshow(edges_image, cmap='gray')
            plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 2, 2), plt.imshow(image, cmap='gray')
            plt.title('HoughLines Transform'), plt.xticks([]), plt.yticks([])
            plt.show()
            return image
        return rotated_image
    else:
        print("NO LINES!")
        return image

def is_grayscale_02(img_PIL):
    state = "Gray Image"  # initial state is gray

    # image is PIL.Image, need to convert to nparray
    image = np.array(img_PIL)
    height, width, _ = image.shape

    # devide image into 9 grid-cell
    grid_size_h = height // 3
    grid_size_w = width // 3

    for i in range(3):
        for j in range(3):
            # get patch
            patch = image[i * grid_size_h:(i + 1) * grid_size_h, j * grid_size_w:(j + 1) * grid_size_w]

            # split channels to R, G, B
            b_channel, g_channel, r_channel = cv2.split(patch)

            # compute the differences between channels
            diff_rg = cv2.absdiff(r_channel, g_channel)
            diff_rb = cv2.absdiff(r_channel, b_channel)
            diff_gb = cv2.absdiff(g_channel, b_channel)

            # Create a mask to determine which pixels have a color based on a threshold
            mask_color = (diff_rg > color_threshold) | (diff_rb > color_threshold) | (diff_gb > color_threshold)

            # compute ratio of coloured pixel in grid-cell
            num_color_pixels = np.sum(mask_color)
            total_pixels = mask_color.size
            color_pixel_ratio = num_color_pixels / total_pixels
            mean_diff_score = (np.mean(diff_rg) + np.mean(diff_rb) + np.mean(diff_gb)) / 3
            # print("Mean score:", mean_diff_score)
            # print("Ratio:", color_pixel_ratio)

            if color_pixel_ratio > pixel_ratio_threshold and mean_diff_score > color_threshold:
                state = "Coloured Image"
                return state
    return state

def load_model_v3(model_path, num_classes):
    print("Loading MobileNetV3...")
    # Khởi tạo mô hình MobileNetV3 với số lượng class cụ thể
    model = models.mobilenet_v3_large()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)

    # Load trọng số mô hình đã huấn luyện
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Chuyển mô hình sang chế độ đánh giá (evaluation)

    return model

def load_model_iqa():
    # set up IQA model
    print("Loading IQA model...")
    iqa_model = create_metric("brisque", metric_mode="NR")
    return iqa_model

def predict_with_threshold(model, image_converted):
    # Chuyển từ cv2 sang PIL
    # color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(color_coverted)

    image = image_converted.convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Thêm batch dimension

    # Thực hiện dự đoán
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        # print(probabilities)

        # Kiểm tra xem xác suất lớn nhất có vượt qua ngưỡng không
        if max_prob.item() < threshold_cls:
            return "Unknown"
        predicted_class = class_names[predicted.item()]

    return predicted_class

def inference(page_num, image, model_cls_1, model_cls_2, model_det, model_iqa, path_page):
    # Resize image if needed
    flag, image = is_resize(image)  # image to inference

    if not flag:
        print("Image size is too small. Ignore image...!!")
        return

    # Start Inference
    # Classify 3 types of document
    predicted_class = predict_with_threshold(model_cls_1, image)
    color_state = is_grayscale_02(img_PIL=image)
    print("Predicted Class Done!")

    if predicted_class == "card":
        print("Predicted class:", predicted_class)

        image_tensor = transform(image).unsqueeze(0).to(device)
        print("Starting Detect Card!")
        with torch.no_grad():
            prediction = model_det(image_tensor)
            # print(prediction[0])
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        # # Reject if more than 1 card in image
        # if len(boxes) >= 2:
        #     print("More than 1 card in image!")
        #     return page_payload

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score > threshold_det:
                card_data = dict()

                print("__det_score__:", score, end="  |  ")
                x_min, y_min, x_max, y_max = map(int, box)
                # width = x_max - x_min
                # height = y_max - y_min

                crop_card = np.array(image)[y_min:y_max, x_min:x_max]

                # Rotate if needed
                crop_card = rotate(crop_card)

                # Image quality assessment
                iqa_score = model_iqa(Image.fromarray(crop_card), None).cpu().item()
                print("quality_score:", round(iqa_score, 2), end="  |  ")

                card_path = path_page + f"/card_{i+1}.jpg"

                if iqa_score > 40:
                    print("bad image")
                    print("IQA Done!")
                    Image.fromarray(crop_card).save(card_path)
                    print(f"Saved crop card to {card_path}\n")
                    continue
                print("good image")
                print("IQA Done!")

                if color_state == "gray image":
                    print("Type card and Gray image --> Pass!")
                else:
                    predict_class_v2 = predict_with_threshold(model_cls_2, Image.fromarray(crop_card))
                    print("Predicted Crop Image Done!")
                    if predict_class_v2 != "card":
                        print("Type of Detected area != Type of Image")
                        continue

                Image.fromarray(crop_card).save(card_path)
                print(f"Saved crop card to {card_path}\n")
            else:
                # page_payload["comment"] = "not found card!"
                print("Classify to card but not found card!")
        print("Detect Done!")

        print("Finish Process !!!")
        return

    elif predicted_class == "administrative_document":
        print("Predicted class:", predicted_class)

        # Image quality assessment
        iqa_score = model_iqa(image, None).cpu().item()
        print("quality_score:", round(iqa_score, 2), end="  |  ")
        if iqa_score > 85:
            print("bad image")
        else:
            print("good image")
        print("IQA Done!")

        page_path = path_page + f"/administrative_document_page_{page_num+1}.jpg"
        image.save(page_path)
        print(f"Saved image to {page_path}\n")
    elif predicted_class == "face":
        print("Predicted class:", predicted_class)

        # Image quality assessment
        iqa_score = model_iqa(image, None).cpu().item()
        print("quality_score:", round(iqa_score, 2), end="  |  ")
        if iqa_score > 50:
            print("bad image")
        else:
            print("good image")
        print("IQA Done!")

        page_path = path_page + f"/face_page_{page_num + 1}.jpg"
        image.save(page_path)
        print(f"Saved image to {page_path}\n")
    else:
        print("Predicted class:", predicted_class)
    return


if __name__ == "__main__":
    image_dir = "./image_test"

    model_cls1 = load_model_v3("mobilenetv3_Large_add_Grayscale.pth", num_classes=len(class_names))
    model_cls2 = load_model_v3("mobilenetv3_Large.pth", num_classes=len(class_names))

    print("Loading Faster RCNN...")
    model_det = torch.load("./CCCD_model_Clean_Data_80e.pth")
    model_det.eval()
    model_det.to(device)

    model_iqa = load_model_iqa()

    count_file = 0
    request_code = "AI-000"

    start_time = time.time()
    for file in os.listdir(image_dir):
        count_file += 1
        request_code = request_code + str(count_file)
        file_extension = file.split('.')[-1].lower()
        # Tạo file_id duy nhất
        file_id = str(uuid.uuid4())
        # Tạo thư mục lưu trữ nếu nó chưa tồn tại
        path_by_request_code = f"{FILEPATH}/{request_code}"
        Path(path_by_request_code).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(path_by_request_code, f"{file_id}.{file_extension}")
        # Lưu file với tên là file_id
        with open(os.path.join(image_dir, file), "rb") as fh:
            file_content = fh.read()
        with open(file_path, "wb") as f:
            f.write(file_content)

        print("\n//////////////////////////////")
        if file.endswith(".pdf"):
            pages = pdf2image.convert_from_path(os.path.join(image_dir, file))
            # Loop for pages
            for i, page in enumerate(pages):
                path_page = f"{FILEPATH}/{request_code}/page_{str(i + 1)}"
                Path(path_page).mkdir(parents=True, exist_ok=True)
                inference(i, page, model_cls1, model_cls2, model_det, model_iqa, path_page)

        elif file.endswith((".jpg", ".png")):
            page_num = 0
            path_page = f"{FILEPATH}/{request_code}/page_{str(page_num + 1)}"
            Path(path_page).mkdir(parents=True, exist_ok=True)
            image = Image.open(os.path.join(image_dir, file))
            inference(page_num, image, model_cls1, model_cls2, model_det, model_iqa, path_page)

    print("Avg time:", (time.time() - start_time)/count_file)
