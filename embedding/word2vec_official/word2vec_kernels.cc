/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h" // 并行生成随机数
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

// Number of examples to precalculate.
const int kPrecalc = 3000;
// Number of words to read into a sentence before processing.
const int kSentenceSize = 1000;

namespace {

bool ScanWord(StringPiece* input, string* word) {
  str_util::RemoveLeadingWhitespace(input);
  StringPiece tmp;
  if (str_util::ConsumeNonWhitespace(input, &tmp)) {
    word->assign(tmp.data(), tmp.size());
    return true;
  } else {
    return false;
  }
}

}  // end namespace
// 每次处理一个batch

class SkipgramWord2vecOp : public OpKernel {
 public:
  explicit SkipgramWord2vecOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), rng_(&philox_) {
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
    // 注意此处调用了初始化程序
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename)); // 调用初始化程序

    mutex_lock l(mu_);
    example_pos_ = corpus_size_; // example_pos_是指在语料库中的索引
    label_pos_ = corpus_size_;
    label_limit_ = corpus_size_; // 不允许超出语料库的大小
    sentence_index_ = kSentenceSize; // 一个句子中最大的单词个数1000
    for (int i = 0; i < kPrecalc; ++i) {　// 事先处理3000个word
      NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
    }
  }

  // 每次处理一个batch
  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({batch_size_})); // examples声明为batch_size大小的Tensor
    auto Texamples = examples.flat<int32>(); // 将examples展开为一维向量
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>(); // 一维的向量
    {
      mutex_lock l(mu_);
      // 将vector里一个batch的数据读到Tensor中去
      for (int i = 0; i < batch_size_; ++i) {
        Texamples(i) = precalc_examples_[precalc_index_].input; // vector<Example>
        Tlabels(i) = precalc_examples_[precalc_index_].label;
        precalc_index_++;
        // 每当预计算的样本读取完毕时，我们就重新读取一批
        if (precalc_index_ >= kPrecalc) {
          precalc_index_ = 0;
          for (int j = 0; j < kPrecalc; ++j) {
            NextExample(&precalc_examples_[j].input,
                        &precalc_examples_[j].label);
          }
        }
      }
      words_per_epoch.scalar<int64>()() = corpus_size_; // 将tensor转为一个标量
      current_epoch.scalar<int32>()() = current_epoch_; // 当前的epoch
      total_words_processed.scalar<int64>()() = total_words_processed_; //处理了多少单词
    }
    // 将tensor设置为函数输出
    ctx->set_output(0, word_);//word_ 是一维的string Tensor
    ctx->set_output(1, freq_);// freq_ 是一维的int32 Tensor
    ctx->set_output(2, words_per_epoch);// 每个epoch处理多少个单词
    ctx->set_output(3, current_epoch); // 当前的epoch
    ctx->set_output(4, total_words_processed); // 当前已经处理了多少个单词
    ctx->set_output(5, examples); // 一个batch 的样本 上下文词index
    ctx->set_output(6, labels); // 一个batch的样本的 目标词的index
  }

 private:
  struct Example {
    int32 input;
    int32 label;
  };

  int32 batch_size_ = 0;
  int32 window_size_ = 5;
  float subsample_ = 1e-3;
  int min_count_ = 5;
  int32 vocab_size_ = 0;
  Tensor word_;
  Tensor freq_;
  int64 corpus_size_ = 0;
  std::vector<int32> corpus_; // 将语料中的每个词转成index存起来
  std::vector<Example> precalc_examples_; // 大小为3000
  int precalc_index_ = 0;
  std::vector<int32> sentence_;
  int sentence_index_ = 0;

  mutex mu_;
  random::PhiloxRandom philox_ GUARDED_BY(mu_);
  random::SimplePhilox rng_ GUARDED_BY(mu_);
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int64 total_words_processed_ GUARDED_BY(mu_) = 0;
  int64 example_pos_ GUARDED_BY(mu_);
  int32 label_pos_ GUARDED_BY(mu_);
  int32 label_limit_ GUARDED_BY(mu_);

  // {example_pos_, label_pos_} is the cursor for the next example.
  // example_pos_ wraps around at the end of corpus_. For each
  // example, we randomly generate [label_pos_, label_limit) for
  // labels.
  void NextExample(int32* example, int32* label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    while (true) {
      if (label_pos_ >= label_limit_) { // label_limit 为语料库的大小
        ++total_words_processed_; // 总共处理了多少个单词
        ++sentence_index_; // 句子里的index加1
        if (sentence_index_ >= kSentenceSize) { // 超出句子最大长度
          sentence_index_ = 0;
          for (int i = 0; i < kSentenceSize; ++i, ++example_pos_) {// i是句子中的索引
            if (example_pos_ >= corpus_size_) { // example_pos_是指在语料库中的索引
              ++current_epoch_; // 一个epoch处理完成
              example_pos_ = 0; // 语料库从头开始采样
            }
            // 对语料中的词按概率进行丢弃
            if (subsample_ > 0) {
              int32 word_freq = freq_.flat<int32>()(corpus_[example_pos_]); //获取某个词的freq,corpus_:vector<int32>
              // See Eq. 5 in http://arxiv.org/abs/1310.4546
              // 根据词频来计算一个词是否需要丢弃，词频越高的词，越容易被丢弃
              float keep_prob =
                  (std::sqrt(word_freq / (subsample_ * corpus_size_)) + 1) *
                  (subsample_ * corpus_size_) / word_freq; // 公式里好像没有corpus_size_
              if (rng_.RandFloat() > keep_prob) { // 以一定概率去掉
                i--; // 该词被丢弃，所以并不算在句子长度之内
                continue;
              }
            }
            sentence_[i] = corpus_[example_pos_];  // 将词放句子中去
          } // for
        }
        // SimplePhilox.Uniform integer in [0, n).
        const int32 skip = 1 + rng_.Uniform(window_size_); // windows_size = 5, 从窗口里选一个步长
        // 但从此处看，都是用中心词来预测其前面的词
        label_pos_ = std::max<int32>(0, sentence_index_ - skip); // 以当前词窗口内的词作为target
        // sentence_index_为中心词，label_pos_ 为待预测的词
        label_limit_ =
            std::min<int32>(kSentenceSize, sentence_index_ + skip + 1);
      } // end of if
      if (sentence_index_ != label_pos_) {
        break; // 如果中心词与预测词不同，那么跳出while,生成一个样本
      }
      ++label_pos_;
    } // end of while
    *example = sentence_[sentence_index_];
    *label = sentence_[label_pos_++];
  }

  Status Init(Env* env, const string& filename) {
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data)); // 将file里的数据读到data里
    StringPiece input = data;
    string w;
    corpus_size_ = 0; // 词料库的大小
    std::unordered_map<string, int32> word_freq; // 词 -> 词频
    while (ScanWord(&input, &w)) {
      ++(word_freq[w]); // c++ map初始始化值均为0
      ++corpus_size_;
    }
    if (corpus_size_ < window_size_ * 10) {
      return errors::InvalidArgument("The text file ", filename,
                                     " contains too little data: ",
                                     corpus_size_, " words");
    }
    typedef std::pair<string, int32> WordFreq;
    std::vector<WordFreq> ordered; // vector里装的都是 word_freq ,去除了那些低频的词
    for (const auto& p : word_freq) {
      if (p.second >= min_count_) ordered.push_back(p); // 只对有效的word_freq 进行
    }
    LOG(INFO) << "Data file: " << filename << " contains " << data.size()
              << " bytes, " << corpus_size_ << " words, " << word_freq.size()
              << " unique words, " << ordered.size() //  去重之后的词个数
              << " unique frequent words.";
    word_freq.clear();
    // 对vector按内容进行排序
    std::sort(ordered.begin(), ordered.end(),
              [](const WordFreq& x, const WordFreq& y) {
                return x.second > y.second;
              }); // 降序排序
    vocab_size_ = static_cast<int32>(1 + ordered.size()); // 词汇表的大小，加上一个未登录词
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    word.flat<string>()(0) = "UNK"; // 一维向量,里面装的是string
    static const int32 kUnkId = 0;
    std::unordered_map<string, int32> word_id; // word -> index
    int64 total_counted = 0;
    for (std::size_t i = 0; i < ordered.size(); ++i) { // 遍历去重之后的词个数
      const auto& w = ordered[i].first; // word -> count
      auto id = i + 1; // 第0个是未登录词
      word.flat<string>()(id) = w; // word
      auto word_count = ordered[i].second;
      freq.flat<int32>()(id) = word_count; // count
      total_counted += word_count;
      word_id[w] = id;
    }
    freq.flat<int32>()(kUnkId) = corpus_size_ - total_counted; // 其它的都看成未登录词
    word_ = word;　// word_ 是一维的string Tensor
    freq_ = freq; // freq_ 是一维的int32 Tensor
    corpus_.reserve(corpus_size_); // vector<int32>
    input = data;
    // corpus_:vector<int32>
    while (ScanWord(&input, &w)) { // string w ;
      corpus_.push_back(gtl::FindWithDefault(word_id, w, kUnkId)); // 从word_id中查找词w的index,若未找到则返回未登录词的index=0
    }
    precalc_examples_.resize(kPrecalc); //3000
    sentence_.resize(kSentenceSize); //1000
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("SkipgramWord2vec").Device(DEVICE_CPU), SkipgramWord2vecOp);

class NegTrainWord2vecOp : public OpKernel {
 public:
  explicit NegTrainWord2vecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    base_.Init(0, 0);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));

    std::vector<int32> vocab_count;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_count", &vocab_count));

    std::vector<float> vocab_weights;
    vocab_weights.reserve(vocab_count.size());
    for (const auto& f : vocab_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights.push_back(r);
    }
    sampler_ = new random::DistributionSampler(vocab_weights);
  }

  ~NegTrainWord2vecOp() { delete sampler_; }

  void Compute(OpKernelContext* ctx) override {
    Tensor w_in = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()),
                errors::InvalidArgument("Must be a matrix"));
    Tensor w_out = ctx->mutable_input(1, false);
    OP_REQUIRES(ctx, w_in.shape() == w_out.shape(),
                errors::InvalidArgument("w_in.shape == w_out.shape"));
    const Tensor& examples = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()),
                errors::InvalidArgument("Must be a vector"));
    const Tensor& labels = ctx->input(3);
    OP_REQUIRES(ctx, examples.shape() == labels.shape(),
                errors::InvalidArgument("examples.shape == labels.shape"));
    const Tensor& learning_rate = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()),
                errors::InvalidArgument("Must be a scalar"));

    auto Tw_in = w_in.matrix<float>();
    auto Tw_out = w_out.matrix<float>();
    auto Texamples = examples.flat<int32>();
    auto Tlabels = labels.flat<int32>();
    auto lr = learning_rate.scalar<float>()();
    const int64 vocab_size = w_in.dim_size(0);
    const int64 dims = w_in.dim_size(1);
    const int64 batch_size = examples.dim_size(0);
    OP_REQUIRES(ctx, vocab_size == sampler_->num(),
                errors::InvalidArgument("vocab_size mismatches: ", vocab_size,
                                        " vs. ", sampler_->num()));

    // Gradient accumulator for v_in.
    Tensor buf(DT_FLOAT, TensorShape({dims}));
    auto Tbuf = buf.flat<float>();

    // Scalar buffer to hold sigmoid(+/- dot).
    Tensor g_buf(DT_FLOAT, TensorShape({}));
    auto g = g_buf.scalar<float>();

    // The following loop needs 2 random 32-bit values per negative
    // sample.  We reserve 8 values per sample just in case the
    // underlying implementation changes.
    auto rnd = base_.ReserveSamples32(batch_size * num_samples_ * 8);
    random::SimplePhilox srnd(&rnd);

    for (int64 i = 0; i < batch_size; ++i) {
      const int32 example = Texamples(i);
      DCHECK(0 <= example && example < vocab_size) << example;
      const int32 label = Tlabels(i);
      DCHECK(0 <= label && label < vocab_size) << label;
      auto v_in = Tw_in.chip<0>(example);

      // Positive: example predicts label.
      //   forward: x = v_in' * v_out
      //            l = log(sigmoid(x))
      //   backward: dl/dx = g = sigmoid(-x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      {
        auto v_out = Tw_out.chip<0>(label);
        auto dot = (v_in * v_out).sum(); // 应该还有一个bias 才对
        g = (dot.exp() + 1.f).inverse();
        Tbuf = v_out * (g() * lr);
        v_out += v_in * (g() * lr);
      }

      // Negative samples:
      //   forward: x = v_in' * v_sample
      //            l = log(sigmoid(-x))
      //   backward: dl/dx = g = -sigmoid(x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      for (int j = 0; j < num_samples_; ++j) {
        const int sample = sampler_->Sample(&srnd);
        if (sample == label) continue;  // Skip.
        auto v_sample = Tw_out.chip<0>(sample);
        auto dot = (v_in * v_sample).sum();
        g = -((-dot).exp() + 1.f).inverse();
        Tbuf += v_sample * (g() * lr);
        v_sample += v_in * (g() * lr);
      }

      // Applies the gradient on v_in.
      v_in += Tbuf;
    }
  }

 private:
  int32 num_samples_ = 0;
  random::DistributionSampler* sampler_ = nullptr;
  GuardedPhiloxRandom base_;
};

REGISTER_KERNEL_BUILDER(Name("NegTrainWord2vec").Device(DEVICE_CPU), NegTrainWord2vecOp);

}  // end namespace tensorflow
