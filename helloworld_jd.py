# rewriting the tensorflow2.0 notebook examples as python files

import tensorflow as tf
#dummy comment
hello = tf.constant("hello, world! it's me, the tensor")
print("Here's the tensor in raw form.")
print(hello)
print('Here''s the tensor value accessed via numpy')
#tensorflow stores strings as byte arrays. need to decode w/ utf8 when printing value
print(hello.numpy().decode('utf-8'))
