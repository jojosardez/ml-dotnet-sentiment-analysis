﻿using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SentimentAnalysis
{
    class Program
    {
        const string _dataPath = @".\Data\wikipedia-detox-250-line-data.tsv";
        const string _testDataPath = @".\Data\wikipedia-detox-250-line-test.tsv";

        static void Main(string[] args)
        {
            Console.WriteLine("Sentiment analysis using ML.NET");
            Console.WriteLine();
            Console.WriteLine("Sample positive comment: \"He is the best, and the article should say that.\"");
            Console.WriteLine("Sample negative comment: \"Please refrain from adding nonsense to Wikipedia.\"");
            Console.WriteLine();
            
            while (true)
            {
                Console.WriteLine("Type in your comment or type in 'exit' and press [ENTER]:");
                var input = Console.ReadLine()?.Trim();

                if (string.IsNullOrEmpty(input))
                {
                    Console.WriteLine("Invalid input. Please try again.");
                    continue;
                }

                if (string.Equals(input, "exit", StringComparison.OrdinalIgnoreCase))
                    break;
                else
                {
                    var model = TrainAndPredict(input);
                    Evaluate(model);
                    Console.WriteLine();
                    Console.WriteLine();
                }
            }
        }

        public static PredictionModel<SentimentData, SentimentPrediction> TrainAndPredict(string comment)
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

            // Data
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = comment
                }
            };

            // Prediction
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            // Display prediction / sentiment
            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            // Evaluate model
            var testData = new TextLoader<SentimentData>(_testDataPath, useHeader: true, separator: "tab");
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            // Display evaluation
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
    }
}
