import numpy.linalg.linalg
import numpy as np
import torch
import sys
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import random


def read_data(path):
    images = os.listdir(path)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
    data = {}

    for k in range(len(images)):
        #print(images[k])
        input_image = Image.open(path + "/" + images[k])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output[0])
        data[k] = output[0]
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        # probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # print(probabilities)

    return data


def cosine_dist(d1, d2):
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    dist = np.dot(d1, d2) / (n1 * n2)
    return abs(1 - dist)


def k_medoids(data, medoids):
    old_cost = 100
    clusters, new_cost = return_clusters(data, medoids)
    iteration = 0
    # print("iteration: ", iteration, ", cost: ", new_cost)
    while old_cost > new_cost:
        old_cost = new_cost
        medoids = recalculate_medoids(clusters, data)
        clusters, new_cost = return_clusters(data, medoids)
        # print_clusters(indexes)
        iteration += 1
        # print("iteration: ", iteration, ", cost: ", new_cost)

    return list(clusters.values())


def return_clusters(data, medoids):
    clusters = {}
    cost = 0
    distances = {}
    # print(medoids)
    for j in medoids:
        clusters[j] = []

    for i in data:
        min_dist = 1
        closest_medoid = None
        for j in medoids:
            j_dist = cosine_dist(data[i], data[j])
            if j_dist < min_dist:
                min_dist = j_dist
                closest_medoid = j

        # print(i, closest_medoid)
        clusters[closest_medoid].append(i)
        cost += min_dist
    # print(clusters)
    return clusters, cost


def recalculate_medoids(clusters, data):
    new_medoids = []

    for i in clusters:
        min_dist_sum = 100
        new_medoid = -1
        for medoid_candidate in clusters[i]:
            cur_dist_sum = dist_sum(clusters[i], data[medoid_candidate], data)
            # print(cur_dist_sum)
            if cur_dist_sum < min_dist_sum:
                min_dist_sum = cur_dist_sum
                new_medoid = medoid_candidate
        # print("\n")
        new_medoids.append(new_medoid)

    return new_medoids


def print_clusters(indexes):
    for el in indexes:
        i = 0
        #print(el)


def dist_sum(list, point, data):
    sum = 0
    for el in list:
        sum += cosine_dist(data[el], point)

    return sum


def silhouette(el, clusters, data):
    a = 0
    oth_clusters = []
    for cluster in clusters:
        if el in cluster:
            if len(cluster) == 1:
                return 0

            sum = 0
            for point in cluster:
                sum += cosine_dist(data[el], data[point])
            a = (1 / (len(cluster) - 1)) * sum

        else:
            sum = 0
            for point in cluster:
                sum += cosine_dist(data[el], data[point])
            oth_clusters.append((1 / len(cluster)) * sum)

    b = min(oth_clusters)
    s = (b - a) / max(a, b)

    return s


def silhouette_average(data, clusters):
    sum = 0
    count = 0
    for cluster in clusters:
        for el in cluster:
            count += 1
            sum += silhouette(el, clusters, data)
            # print(el, silhouette(el, clusters, data))
    av = sum / count

    return av


if __name__ == "__main__":
    if len(sys.argv) == 3:
        K = sys.argv[1]
        path = sys.argv[2]

    else:
        K = 5
        path = "stock_images"

    K = int(K)
    #print(K, path, type(K))
    data = read_data(path)
    costs_clusters = {}

    for i in range(100):
        medoids = random.sample(list(data), K)
        clusters = k_medoids(data, medoids)
        av = silhouette_average(data, clusters)
        costs_clusters[av] = clusters
        #print("i: ", i, "avg_silh: ", av)

    max_av = max(costs_clusters)
    best_clustering = costs_clusters[max_av]
    #print(max_av)
    #print(best_clustering)

    images = os.listdir(path)

    k = 1
    for cluster in best_clustering:
        silh_dict = {}
        height = 200
        width = 100 * len(cluster)
        canvas = Image.new('RGB', (width, height))
        for el in cluster:
            silh_dict[silhouette(el, best_clustering, data)] = el

        i = 0
        for silh in sorted(silh_dict.keys()):
            img = Image.open(path + "/" + images[silh_dict[silh]])
            img = img.resize((100, 60))
            canvas.paste(img, (int(width * i / len(cluster)), int(height / 2)))
            font = ImageFont.load_default()
            draw = ImageDraw.Draw(canvas)
            draw.text((int(width * i / len(cluster)), int(height / 4)),
                      str(round(silh, 2)), font=font)
            i += 1

        canvas.save("cluster_" + str(round(k, 2)) + ".PNG", "PNG")
        k += 1

