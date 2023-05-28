
class Dataset_Image():

    def __init__(self, name, path, tracklet, id, camera, frame):
        self.name = name
        self.path = path
        self.tracklet = tracklet
        self.id = id
        self.camera = camera
        self.frame = frame
        self.uidt = str(id).zfill(3) + str(tracklet).zfill(3)
    def __str__(self):
        return self.path
    def __eq__(self, other):
        if isinstance(other, Dataset_Image):
            return self.path == other.path
        return False
    def __hash__(self):
        return hash(self.path)
    def get_individual(self):
        return tuple([self.id, self.tracklet])