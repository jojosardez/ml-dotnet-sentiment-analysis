using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace SentimentAnalysis
{
    class Program
    {
        const string _dataPath = @".\Data\wikipedia-detox-250-line-data.tsv";
        const string _testDataPath = @".\Data\wikipedia-detox-250-line-test.tsv";

        static void Main(string[] args)
        {
            var model = TrainAndPredict();
        }

        public static PredictionModel<SentimentData, SentimentPrediction> TrainAndPredict()
        {
            // Ingest the data
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<SentimentData>(_dataPath, useHeader: true, separator: "tab"));

            // Data preprocess and feature engineering
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            // Model classification
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            // Train the model
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            return model;
        }
    }
}
