import numpy as np
import random


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))
        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
        sampled_faces = (random.choices(faces,
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        sampled_points = np.zeros((self.output_size, 3))
        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        return sampled_points

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()
    return fig


def DeepSobel(im): #Apply sobel oprator of response map of specific layer in the net to get its total gradient/edge map of this layer
    im=im.squeeze()#Remove acces dimension
    im=np.swapaxes(im,0,2)#Swap axes to feet thos of standart image (x,y,d)
    im=np.swapaxes(im,1,2)
    Gx=[[1,2,1], [0 , 0 ,0],[-1,-2,-1]]#Build sobel x,y gradient filters
    Gy=np.swapaxes(Gx,0,1)#Build sobel x,y gradient filters
    ndim=im[:,1,1].shape[0]# Get the depth (number of filter of the layer)
    TotGrad=np.zeros(im[1,:,:].shape) #The averge gradient map of the image to be filled later

    for ii in range(ndim):# Go over all dimensions (filters)

        gradx = signal.convolve2d(im[ii,:,:],Gx,  boundary='symm',mode='same');#Get x sobel response of ii layer
        grady = signal.convolve2d(im[ii,:,:],Gy,  boundary='symm',mode='same');#Get y sobel response of ii layer
        grad=np.sqrt(np.power(gradx,2)+np.power(grady,2));#Get total sobel response of ii layer
        TotGrad+=grad#Add to the layer average gradient/edge map
    TotGrad/=ndim#Get layer sobel gradient map
    return TotGrad
