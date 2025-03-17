import tensorflow as tf

class EmbeddingSlotData(object):
    def __init__(self) -> None:
        self.tot_dim = 0
        self.slot_slice_map = {}  # key: slot_str, value: (begin_ind, len)
        self.slot_count_map = {}  # key: slot_id, value: count


class FeatureImportanceUtil:
    embedding_slots_data_map = {}
    extra_slots_data_map = {}  # key: slot_str, value: embedding
    config = None

    def __init__(self, config):
        self.config = config

    def config_new_embedding(self, *args, **kwargs):
        name = args[0]
        slots = kwargs["slots"]
        dim = kwargs["dim"]
        embedding_slots_data = EmbeddingSlotData()
        if isinstance(slots, str):
            slots = slots.split(" ")
        for slot_id in slots:
            # update slot appearance count
            slot_key = "{}".format(slot_id)
            count = embedding_slots_data.slot_count_map.get(slot_id, 0)
            if count > 0:
                slot_key = "{}_{}".format(slot_id, count)
            embedding_slots_data.slot_count_map[slot_id] = count + 1
            # udpate slot embedding position info
            embedding_slots_data.slot_slice_map[slot_key] = (
                embedding_slots_data.tot_dim,
                dim,
            )
            embedding_slots_data.tot_dim += dim

        self.embedding_slots_data_map[name] = embedding_slots_data
        return self.config.new_embedding(*args, **kwargs)


    def calculate(self, loss, extra_slots_data_map=dict()):
        importance_factor_list = []
        slot_id_list = []
        print("loss shape: {}".format(loss.shape))
        for key, config_embedding in self.config._cached_embedding_variable.items():
            print("{} shape: {}".format(key, config_embedding.shape))
            gradient = tf.gradients(loss, config_embedding)
            if not (len(gradient) > 0 and tf.contrib.framework.is_tensor(gradient[0])):
                print(
                    "{} has no gradient in loss with len {}".format(key, len(gradient))
                )
                continue
            sub_importance_factor = tf.nn.sigmoid(gradient)
            print("{} gradient shape: {}".format(key, sub_importance_factor.shape))
            embedding_slots_data = self.embedding_slots_data_map[key]
            for slot_id_str, (
                begin_ind,
                dim,
            ) in embedding_slots_data.slot_slice_map.items():
                importance_factor = sub_importance_factor[:,:,begin_ind:begin_ind+dim]
                # importance_factor = sub_importance_factor
                print("{}: begin_ind: {}, dim: {} shape: {}".format(slot_id_str, begin_ind, dim, importance_factor.shape)) #TODO: delete
                tf.summary.histogram('histogram_input_gradients/{}'.format(slot_id_str), importance_factor)

                scalar_importance_factor = tf.reshape(
                    importance_factor,
                    [
                        -1,
                    ],
                )
                _, scalar_importance_factor = tf.nn.moments(
                    scalar_importance_factor, axes=[0]
                )
                scalar_importance_factor = tf.sqrt(scalar_importance_factor)
                # print(scalar_importance_factor.shape)
                tf.summary.scalar(
                    "scalar_input_gradients_std/{}".format(slot_id_str),
                    scalar_importance_factor,
                )
                importance_factor_list.append(scalar_importance_factor)
                slot_id = slot_id_str.split("_")[0]
                slot_id_list.append(int(slot_id))
                break

        for slot_id_str, config_embedding in extra_slots_data_map.items():
            gradient = tf.gradients(loss, config_embedding)
            importance_factor = tf.nn.sigmoid(gradient)
            #   tf.summary.histogram('histogram_input_gradients/{}'.format(slot_id_str), importance_factor)
            scalar_importance_factor = tf.reshape(
                importance_factor,
                [
                    -1,
                ],
            )
            _, scalar_importance_factor = tf.nn.moments(
                scalar_importance_factor, axes=[0]
            )
            scalar_importance_factor = tf.sqrt(scalar_importance_factor)
            # print(scalar_importance_factor.shape)
            tf.summary.scalar(
                "scalar_input_gradients_std/{}".format(slot_id_str),
                scalar_importance_factor,
            )
            importance_factor_list.append(scalar_importance_factor)
            slot_id = slot_id_str.split("_")[0]
            slot_id_list.append(int(slot_id))

        slot_id_tensor = tf.constant(slot_id_list, tf.float32)
        slot_id_tensor = tf.reshape(slot_id_tensor, [-1, 1])
        print("slot_id_tensor shape: {}".format(slot_id_tensor.shape))

        importance_factor_tensor = tf.stack(importance_factor_list, axis=0)
        importance_factor_tensor = tf.reshape(importance_factor_tensor, [-1, 1])
        print("importance_factor_tensor: {}".format(importance_factor_tensor.shape))

        importance_table = tf.concat(
            [importance_factor_tensor, slot_id_tensor], axis=1
        )
        sort_ind = tf.contrib.framework.argsort(importance_table, axis=0, direction="DESCENDING")
        sort_importance_table = tf.gather_nd(importance_table, sort_ind[:, 0:1])
        print("sort_importance_table: {}".format(sort_importance_table.shape))
        tf.summary.text(
            "importance_table", tf.strings.format("{}", sort_importance_table, summarize=-1)
        )
