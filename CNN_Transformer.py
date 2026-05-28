import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

# 2. 改进的Transformer模型
class RoughnessPredictorCNNTransformer(keras.Model):
    def __init__(self,
                 input_shape=(1024, 1),
                 num_heads=4,
                 embed_dim=32,
                 ff_dim=128,
                 num_layers=2,
                 dropout_rate=0.1):
        super().__init__()
        self.input_shape_ = input_shape

        # 1. CNN 前端（局部特征提取）
        self.cnn_frontend = keras.Sequential([
            layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(embed_dim, kernel_size=3, padding="same", activation="relu"),
            # layers.Permute((2, 1))  # [batch, seq_len_reduced, embed_dim]
        ])

        # 计算经过CNN后的序列长度
        self.seq_len_reduced = input_shape[0] // 4  # 经过conv(stride=2)和pooling(pool=2)

        # 2. 位置编码（适配新序列长度）
        self.position_embedding = LearnedPositionalEmbedding(
            max_seq_len=self.seq_len_reduced,
            embed_dim=embed_dim
        )

        # 3. Transformer编码器堆叠
        self.encoders = [
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ]

        # 4. 回归输出头
        self.regression_head = keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dropout(dropout_rate),
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1)  # 回归任务单输出
        ])

    def call(self, inputs, training=None):
        # 1. CNN处理
        x = self.cnn_frontend(inputs)  # [batch, seq_len_reduced, embed_dim]

        # 2. 添加位置信息
        x = self.position_embedding(x)

        # 3. 通过Transformer编码器
        for encoder in self.encoders:
            x = encoder(x, training=training)

        # 4. 回归预测
        return self.regression_head(x)




class LearnedPositionalEmbedding(layers.Layer):
    """可学习的位置编码（适配 CNN-Transformer）"""

    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        # 可学习的位置编码矩阵 [max_seq_len, embed_dim]
        self.pos_emb = layers.Embedding(
            input_dim=max_seq_len,
            output_dim=embed_dim
        )

    def call(self, x):
        """
        输入 x: [batch, seq_len, embed_dim]（来自 CNN 的输出）
        输出: [batch, seq_len, embed_dim]（添加位置编码后的特征）
        """
        seq_len = tf.shape(x)[1]  # 动态获取当前序列长度
        positions = tf.range(start=0, limit=seq_len, delta=1)  # [0, 1, ..., seq_len-1]

        # 获取位置编码并扩展维度以匹配输入 x
        positions = self.pos_emb(positions)  # [seq_len, embed_dim]
        positions = tf.expand_dims(positions, axis=0)  # [1, seq_len, embed_dim]（支持广播）

        return x + positions  # 广播相加 [batch, seq_len, embed_dim]

class TransformerEncoderBlock(layers.Layer):
    """改进的编码器块，适合传感器数据"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # 多头注意力
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # 前馈网络
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# 3. 训练流程
def train_roughness_predictor(X_train, X_val, y_train, y_val):

    # 创建模型
    model = RoughnessPredictorCNNTransformer(
        input_shape=(1024, 1),
        num_heads=4,
        embed_dim=32,
        ff_dim=128,
        num_layers=3
    )

    # # 自定义损失函数（可加入MAE和MSE组合）
    # def hybrid_loss(y_true, y_pred):
    #     mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    #     mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    #     return 0.7 * mae + 0.3 * mse

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mean_absolute_percentage_error']
    )

    # 回调函数
    callbacks = [
        # keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# 示例使用
if __name__ == "__main__":
    X_train = np.load('Train_and_test_dataset_for_prediction_models/X_train_10times_std.npy')
    X_test = np.load('Train_and_test_dataset_for_prediction_models/X_test_10_times_std.npy')
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    Y_ = np.load('Train_and_test_dataset_for_prediction_models/Y_10times.npy')
    Y_train = Y_[:550,:]
    Y_test = Y_[550:,:]
    # 训练模型
    model, history = train_roughness_predictor(X_train,X_test, Y_train, Y_test)

    # 测试预测
    predictions = model.predict(X_test)
