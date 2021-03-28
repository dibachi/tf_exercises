import tensorflow as tf 
# dummy comment
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(5.)

add = tf.add(a,b)
sub = tf.subtract(a,b)
mul = tf.multiply(a,b)
div = tf.divide(a,b)

print("add =", add.numpy())
print("sub =", sub.numpy())
print("mul =", mul.numpy())
print("div =", div.numpy())

mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])

print("mean =", mean.numpy())
print("sum =", sum.numpy())

matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[5., 6.], [7., 8.]])

product = tf.matmul(matrix1, matrix2)

print(product)

print(product.numpy())
