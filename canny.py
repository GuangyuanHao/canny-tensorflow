import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def lrelu(x, leak=0.0002, name='lrelu'): #Leaky relu
    return tf.maximum(leak*x, x,name= name)

def conv(input, out_dim=1, ks=1, s=1, padding = 'SAME',value = [0.114, 0.587, 0.299], name='constant_conv'): #Defining conv2d
    # [0.114, 0.587, 0.299] [0.0721, 0.7154, 0.2125]
    with tf.variable_scope(name):
        return slim.conv2d(input, out_dim,ks,s,padding= padding,activation_fn=None,
                                 weights_initializer=tf.constant_initializer(value=value),
                                 biases_initializer=tf.constant_initializer(0))

def rgb2gray(image): # Gray processing
    c = image.get_shape()[3]
    if c==3:
        gray = conv(image,out_dim=1,ks=1,s=1,value = [0.0721, 0.7154, 0.2125], name='gray')
        return gray
    else:
        return image

def smooth(gray): # Smoothing: blurring the image to remove noise
    # smoothing the image with a Gaussian filter to reduce noise and unwanted details and texture
    value = [1.79106361e-08, 5.93118809e-07, 7.22566631e-06,
             3.23831897e-05, 5.33908537e-05, 3.23831897e-05,
             7.22566631e-06, 5.93118809e-07, 1.79106361e-08,
             5.93118809e-07, 1.96413974e-05, 2.39281205e-04,
             1.07238396e-03, 1.76806225e-03, 1.07238396e-03,
             2.39281205e-04, 1.96413974e-05, 5.93118809e-07,
             7.22566631e-06, 2.39281205e-04, 2.91504184e-03,
             1.30643112e-02, 2.15394077e-02, 1.30643112e-02,
             2.91504184e-03, 2.39281205e-04, 7.22566631e-06,
             3.23831897e-05, 1.07238396e-03, 1.30643112e-02,
             5.85501805e-02, 9.65329280e-02, 5.85501805e-02,
             1.30643112e-02, 1.07238396e-03, 3.23831897e-05,
             5.33908537e-05, 1.76806225e-03, 2.15394077e-02,
             9.65329280e-02, 1.59155892e-01, 9.65329280e-02,
             2.15394077e-02, 1.76806225e-03, 5.33908537e-05,
             3.23831897e-05, 1.07238396e-03, 1.30643112e-02,
             5.85501805e-02, 9.65329280e-02, 5.85501805e-02,
             1.30643112e-02, 1.07238396e-03, 3.23831897e-05,
             7.22566631e-06, 2.39281205e-04, 2.91504184e-03,
             1.30643112e-02, 2.15394077e-02, 1.30643112e-02,
             2.91504184e-03, 2.39281205e-04, 7.22566631e-06,
             5.93118809e-07, 1.96413974e-05, 2.39281205e-04,
             1.07238396e-03, 1.76806225e-03, 1.07238396e-03,
             2.39281205e-04, 1.96413974e-05, 5.93118809e-07,
             1.79106361e-08, 5.93118809e-07, 7.22566631e-06,
             3.23831897e-05, 5.33908537e-05, 3.23831897e-05,
             7.22566631e-06, 5.93118809e-07, 1.79106361e-08]
    smooth = conv(gray,ks=9, s =1,value= value,padding='SAME', name= 'smooth')
    bleed_over = conv(tf.ones_like(gray), ks=9, s=1, value=value, name='bleed_over')
    smooth = smooth/(bleed_over+tf.ones_like(gray)*np.finfo(float).eps)
    return smooth
#-------------------------------------------------------------------------
# Finding gradient: The edges should be marked where the gradients of image has large magnitudes
def sobel(smooth): # Computing the gradients in the x- and  y-direction of the smoothed image using a gradient operator
    #  called Sobel
    smooth = tf.pad(smooth, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    x = [-1, 0, +1,
         -2, 0, +2,
         -1, 0, +1]
    y = [+1, +2, +1,
          0,  0,  0,
         -1, -2, -1]
    Gy = conv(smooth, ks=3, s=1, value=y, padding='VALID', name='Gy')
    Gx = conv(smooth, ks=3, s=1, value=x, padding='VALID', name='Gx')
    return Gy, Gx

def magnitude(Gy,Gx): # Computing the magnitudes of the gradients
    G = tf.sqrt(tf.add(tf.square(Gy),tf.square(Gx)))
    return G
#-------------------------------------------------------------------------
#Non-maximum suppression: Only local maximum should be make as edges
def which_area(Gy,Gx): # Judging the gradient direction in which area of 8 areas
    one = tf.ones_like(Gy)
    zero = tf.zeros_like(Gy)
    area0 = tf.where(tf.equal(tf.abs(Gy)+tf.abs(Gx),zero),one, zero)
    area1 = tf.where(tf.logical_and(tf.greater_equal(tf.abs(Gx),tf.abs(Gy)),
                                    tf.greater_equal(Gx*Gy,zero)),one, zero)
    area1 =area1 - area0
    area4 = tf.where(tf.logical_and(tf.greater_equal(tf.abs(Gx),tf.abs(Gy)),
                                    tf.less(Gx*Gy,zero)),one, zero)

    area2 = tf.where(tf.logical_and(tf.less(tf.abs(Gx), tf.abs(Gy)),
                                    tf.greater_equal(Gx * Gy, zero)), one, zero)
    area3 = tf.where(tf.logical_and(tf.less(tf.abs(Gx), tf.abs(Gy)),
                                    tf.less(Gx * Gy, zero)), one, zero)

    area = 1*area1+2*area2+3*area3+4*area4+5*area0

    return area

def interpolate(G,Gy,Gx, area): # Computing pairs of interpolated gradients in the gradient direction
    G81 = conv(G, 1, ks=3, s=1, value=[0, 0, 0,
                                        0, 0, 1,
                                        0, 0, 0], name='G81')

    G82 = conv(G, 1, ks=3, s=1, value=[0, 0, 1,
                                        0, 0, 0,
                                        0, 0, 0], name='G82')

    G83 = conv(G, 1, ks=3, s=1, value=[0, 1, 0,
                                        0, 0, 0,
                                        0, 0, 0], name='G83')

    G84 = conv(G, 1, ks=3, s=1, value=[1, 0, 0,
                                        0, 0, 0,
                                        0, 0, 0], name='G84')

    G85 = conv(G, 1, ks=3, s=1, value=[0, 0, 0,
                                        1, 0, 0,
                                        0, 0, 0], name='G85')

    G86 = conv(G, 1, ks=3, s=1, value=[0, 0, 0,
                                        0, 0, 0,
                                        1, 0, 0], name='G86')

    G87 = conv(G, 1, ks=3, s=1, value=[0, 0, 0,
                                        0, 0, 0,
                                        0, 1, 0], name='G87')

    G88 = conv(G, 1, ks=3, s=1, value=[0, 0, 0,
                                        0, 0, 0,
                                        0, 0, 1], name='G88')
    value = [
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1,
    ]


    G8 = conv(G, 8, ks=3, s=1, value=value, name='G8')
    shape = G.get_shape()
    G8max = tf.reshape(tf.reduce_max(G8,axis=3),shape=[-1,int(shape[1]),int(shape[2]),1])

    Wone =tf.ones_like(Gy)
    W1 = tf.div(tf.abs(Gy),tf.abs(Gx))
    W2 = tf.div(tf.abs(Gx),tf.abs(Gy))
    areaone = tf.ones_like(area)
    G8izero = tf.zeros_like(G81)
    # Gy = Gx=0
    Gi0 = tf.where(tf.equal(area, 5 * areaone), G8max, G8izero)
    # 1 5
    Gi1 = tf.where(tf.equal(area, 1 * areaone), (W1 * G82 + (Wone - W1) * G81), G8izero)
    Gi5 = tf.where(tf.equal(area, 1 * areaone), (W1 * G86 + (Wone - W1) * G85), G8izero)
    # 2 6
    Gi2 = tf.where(tf.equal(area, 2 * areaone), (W2 * G82 + (Wone - W2) * G83), G8izero)
    Gi6 = tf.where(tf.equal(area, 2 * areaone), (W2 * G86 + (Wone - W2) * G87), G8izero)
    # 3 7
    Gi3 = tf.where(tf.equal(area, 3 * areaone), (W2 * G84 + (Wone - W2) * G83), G8izero)
    Gi7 = tf.where(tf.equal(area, 3 * areaone), (W2 * G88 + (Wone - W2) * G87), G8izero)
    # 4 8
    Gi4 = tf.where(tf.equal(area, 4 * areaone), (W1 * G84 + (Wone - W1) * G85), G8izero)
    Gi8 = tf.where(tf.equal(area, 4 * areaone), (W1 * G88 + (Wone - W1) * G81), G8izero)
    G1 = Gi0 + Gi1 + Gi2 + Gi3 +Gi4
    G2 = Gi0 + Gi5 + Gi6 + Gi7 +Gi8

    return G1, G2

def compare(G, G1,G2):# Comparing the gradient with a pair of gradients in the gradient direction and if the biggest one is
    # is the gradient, then it is marked as edges.
    zero = tf.zeros_like(G)
    GC = tf.where(tf.logical_and(tf.greater_equal(G,G1),tf.greater_equal(G,G2)),G,zero)
    return GC
#--------------------------------------------------------------------------------
#Edge tracking by hystersis
def line(GC, low, high):# Using double thresholds to get weak edges and strong edges
    one = tf.ones_like(GC)
    zero = tf.zeros_like(GC)
    weak = tf.where(tf.greater_equal(GC,low*one),GC, zero)
    strong = tf.where(tf.greater_equal(GC,high*one),GC, zero)
    return weak, strong

def edge(weak, strong):# Getting the final edges
    strongmean = conv(strong,1,ks=3, s=1, value=[1,1,1,
                                                 1,1,1,
                                                 1,1,1,],name = 'strongmean')
    zero = tf.zeros_like(weak)
    edge = tf.where(tf.greater(strongmean, zero), weak, zero)
    return edge

def biedge(edge):# Marking edges as 1 and others as 0
    one = tf.ones_like(edge)
    zero = tf.zeros_like(edge)
    biedge = tf.where(tf.greater(edge,zero),one, zero)
    return biedge
#----------------------------------------------------------------------------
def canny(image,low, high, reuse=False, name="canny"): # Canny
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        gray = rgb2gray(image)
        smoothed = smooth(gray)
        Gy, Gx = sobel(smoothed)
        G = magnitude(Gy, Gx)
        area = which_area(Gy, Gx)
        G1, G2 = interpolate(G, Gy, Gx, area)
        GC = compare(G, G1, G2)
        weak, strong = line(GC, low, high)
        edges = edge(weak, strong)
        canny = biedge(edges)
        return canny



