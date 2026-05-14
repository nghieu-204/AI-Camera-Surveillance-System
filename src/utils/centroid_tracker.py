import math
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box):
        self.objects[self.next_object_id] = (centroid, box)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []
        for rect in rects:
            startX, startY, endX, endY = rect
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id][0] for obj_id in object_ids]

            # Compute distances
            D = []
            for obj_cent in object_centroids:
                row = []
                for inp_cent in input_centroids:
                    dist = math.hypot(obj_cent[0] - inp_cent[0], obj_cent[1] - inp_cent[1])
                    row.append(dist)
                D.append(row)
            
            used_rows = set()
            used_cols = set()
            
            # Simple greedy assignment
            for _ in range(min(len(D), len(D[0]))):
                min_val = float('inf')
                min_row = -1
                min_col = -1
                for r in range(len(D)):
                    if r in used_rows: continue
                    for c in range(len(D[0])):
                        if c in used_cols: continue
                        if D[r][c] < min_val:
                            min_val = D[r][c]
                            min_row = r
                            min_col = c
                
                if min_val > self.max_distance: 
                    break
                    
                object_id = object_ids[min_row]
                self.objects[object_id] = (input_centroids[min_col], rects[min_col])
                self.disappeared[object_id] = 0
                used_rows.add(min_row)
                used_cols.add(min_col)

            # Check disappeared
            for row in range(len(D)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # Check new objects
            for col in range(len(D[0])):
                if col not in used_cols:
                    self.register(input_centroids[col], rects[col])

        return self.objects
