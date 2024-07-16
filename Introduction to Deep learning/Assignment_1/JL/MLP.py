import pandas as pd
import tensorflow as tf

def full_connect():
    train_data = pd.read_csv("/home/math-tr/lsh/Ariel/data/train_in.csv").values  
    train_label = pd.read_csv("/home/math-tr/lsh/Ariel/data/train_out.csv").values
    test_data = pd.read_csv("/home/math-tr/lsh/Ariel/data/test_in.csv").values  
    test_label = pd.read_csv("/home/math-tr/lsh/Ariel/data/test_out.csv").values

    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32,[None,256])
        y_true = tf.placeholder(tf.int32,[None,1])

    with tf.variable_scope("fc_model"):
        weight = tf.Variable(tf.random_normal([256,10],mean=0.0,stddev=1.0),name="weight")
        bias = tf.Variable(tf.constant(0.0,shape=[10]))
        y_predict = tf.matmul(x,weight) + bias

    with tf.variable_scope("soft_cros"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))

    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

    init_op = tf.global_variables_initializer()

    tf.summary.scalar("losses",loss)
    tf.summary.scalar("accuracy",accuracy)
    tf.summary.histogram("weight",weight)
    tf.summary.histogram("bias",bias)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init_op)
        filewriter = tf.summary.FileWriter("./tmp/summary/test/",graph=sess.graph)

        for i in range(200):
            mnist_x , mnist_y = train_data,train_label
            sess.run(train_op,feed_dict={x:mnist_x,y_true:mnist_y})
            summary = sess.run(merged,feed_dict={x:mnist_x,y_true:mnist_y})
            filewriter.add_summary(summary,i)

            print("%dstep,accuracy%f" % (i, sess.run(accuracy,feed_dict={x:mnist_x,y_true:mnist_y})))
    return None
if __name__ == "__main__":
    full_connect()


