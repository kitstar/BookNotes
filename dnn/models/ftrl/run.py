import tensorflow as tf
import sys

tf.app.flags.DEFINE_string("model_type", "", "which model to run")

# use model_type to dispatch models, after import model type, parse gflags again
def main(_):
    model_type = tf.app.flags.FLAGS.model_type
    if model_type == "dnn":
        import dnn_model
        model = dnn_model.AdsDnnModel()
    elif model_type == "lr":
        import lr_model
        model = lr_model.AdsLrModel()
    elif model_type == "resdnn":
        import resdnn_model
        model = resdnn_model.ResDnnModel()
    else:
        raise NotImplementedError()

    # parse gflags again
    tf.app.flags.FLAGS._parse_flags(sys.argv[1:])
    model.run()

if __name__ == "__main__":
    tf.app.run()
