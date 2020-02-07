using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecurrentNeuralNetwork
{
    public class NeuralNetwork
    {
        #region Data Initialization
        public string tokens = "";
        int vocab_size = 0;
        double[,] Wxh;
        double[,] Whh;
        double[,] Why;


        double[,] bh;
        double[,] by;
        Dictionary<int, string> intToText = new Dictionary<int, string>();
        Dictionary<string, int> textToInt = new Dictionary<string, int>();
        #endregion


        #region HyperParameters
        public double learningRate = 0.05;
        public int hiddenNeuron = 100;
        public int sequalLength = 25;
        #endregion

        #region Layer Initialization
        //Default file. MAKE SURE TO CHANGE THIS LOCATION AND FILE PATH TO YOUR FILE 
        static readonly string textFile = @"G:\Testingfiles\testing.txt";
        #endregion

        #region Derivative Initialization
        #endregion

        public NeuralNetwork()
        {
            if (File.Exists(textFile))
            {
                Utility utility = new Utility();
                // Read entire text file content in one string  
                string text = File.ReadAllText(textFile);
                var charList = text.ToCharArray();
                var uniqueChars = charList.Distinct();
                vocab_size = uniqueChars.Count();

                textToInt = utility.ConvertTexttoInteger(charList);
                intToText = utility.ConvertIntegertoText(charList);

                var charA = utility.RandomZeros(vocab_size, 1);
                //charA[textToInt["a"]] = 1;

                Wxh = utility.RandomValues(hiddenNeuron, vocab_size);
                Whh = utility.RandomValues(hiddenNeuron, hiddenNeuron);
                Why = utility.RandomValues(vocab_size, hiddenNeuron);

                bh = utility.RandomZeros(hiddenNeuron, 1);
                by = utility.RandomZeros(vocab_size, 1);
                //FeedForward();
            }
            //string splitTokens = tokens.Substring();
            //char[] uniqueCharacter=

            // Utility utility = new Utility();
            // utility.Add(, 10);

        }



        //public void FeedForward(int[] input, int[] targets, double[] hiddenPrevious)
        public void FeedForward()
        {
            if (File.Exists(textFile))
            {
                Utility utility = new Utility();
                // Read entire text file content in one string  
                //string text = File.ReadAllText(textFile);
                //var charList = text.ToCharArray();
                //var uniqueChars = charList.Distinct();
                //int vocab_size = uniqueChars.Count();

                //var textToInt = utility.ConvertTexttoInteger(charList);
                //var intToText = utility.ConvertIntegertoText(charList);

                //var charA = utility.RandomZeros(vocab_size, 1);
                ////charA[textToInt["a"]] = 1;

                //double[,] Wxh = utility.RandomValues(hiddenNeuron, vocab_size);
                //double[,] Whh = utility.RandomValues(hiddenNeuron, hiddenNeuron);
                //double[,] Why = utility.RandomValues(vocab_size, hiddenNeuron);

                //double[,] bh = utility.RandomZeros(hiddenNeuron, 1);
                //double[,] by = utility.RandomZeros(vocab_size, 1);

                int[] inputs = new int[25] { 43, 69, 58, 4, 66, 68, 73, 69, 62, 69, 60, 13, 4, 76, 63, 58, 69, 4, 35, 73, 58, 60, 68, 73, 4 };
                int[] targets = new int[25] { 69, 58, 4, 66, 68, 73, 69, 62, 69, 60, 13, 4, 76, 63, 58, 69, 4, 35, 73, 58, 60, 68, 73, 4, 47 };
                var hiddenPrevious = utility.RandomZeros(hiddenNeuron, 1);
                //this.FeedForward(inputs, targets, hiddenPrevious);


                Dictionary<int, double[,]> xs = new Dictionary<int, double[,]>();
                Dictionary<int, double[,]> hs = new Dictionary<int, double[,]>();
                Dictionary<int, double[,]> ys = new Dictionary<int, double[,]>();

                Dictionary<int, double[,]> propablity = new Dictionary<int, double[,]>();

                double loss = 0;

                //double[][] hsprevious = new double[hiddenNeuron][];
                //hsprevious = utility.RandomZeros(100, 1);

                //feed forward
                hs[-1] = hiddenPrevious;
                for (int i = 0; i < inputs.Length; i++)
                {
                    try
                    {
                        xs[i] = (utility.RandomZeros(vocab_size, 1));
                        xs[i][inputs[i], 0] = 1;
                        hs[i] = utility.Tanh(utility.Add(utility.Add(utility.MatrixMutiply(Wxh, xs[i]), utility.MatrixMutiply(Whh, hs[i - 1])), bh));
                        //ys[i] = utility.MatrixMutiply(Why, hs[i]) + by;
                        ys[i] = utility.Add(utility.MatrixMutiply(Why, hs[i]), by);
                        propablity[i] = utility.Softmax(ys[i]);
                        loss = loss - Math.Log(propablity[i][targets[i], 0]);
                        //bs[i] = utility.Softmax(ys[i]);
                    }
                    catch (Exception ex)
                    {

                    }


                }

                double[,] dWxh = utility.RandomZeros(hiddenNeuron, vocab_size); ;
                double[,] dWhh = utility.RandomZeros(hiddenNeuron, hiddenNeuron); ;
                double[,] dWhy = utility.RandomZeros(vocab_size, hiddenNeuron); ;
                double[,] dbh = utility.RandomZeros(hiddenNeuron, 1);
                double[,] dby = utility.RandomZeros(vocab_size, 1);
                var dhnext = utility.RandomZeros(hs[0].Length, 1);

                //back propagate
                for (int j = inputs.Length - 1; j > 0; j--)
                {
                    try
                    {
                        var dY = propablity[j];
                        dY[targets[j], 0] -= 1;
                        dWhy = utility.Add(utility.MatrixMutiply(dY, utility.TransposeValues(hs[j])), dWhy);
                        dby = utility.Add(dby, by);
                        var dh = utility.Add(utility.MatrixMutiply(utility.TransposeValues(Why), dY), dhnext);
                        var dhraw = utility.Mutiply(utility.Substract(1, utility.Mutiply(hs[j], hs[j])), dh);
                        dbh = utility.Add(dhraw, dbh);
                        dWxh = utility.Add(utility.MatrixMutiply(dhraw, utility.TransposeValues(xs[j])), dWxh);
                        dWhh = utility.Add(utility.MatrixMutiply(dhraw, utility.TransposeValues(hs[j - 1])), dWhh);
                        dhnext = utility.MatrixMutiply(utility.TransposeValues(Whh), dhraw);

                    }
                    catch (Exception ex)
                    {

                    }


                }




            }





        }




        public void CreateInputs(string text)
        {
            string[] SplitText = text.Split(' ');
            for (int i = 0; i < SplitText.Length; i++)
            {

            }
        }


        public void Train()
        {
            int n = 0;
            double loss = 0;
            double smoothLoss = 0;

            while (n < 100000)
            {
                int[] inputs = new int[25] { 43, 69, 58, 4, 66, 68, 73, 69, 62, 69, 60, 13, 4, 76, 63, 58, 69, 4, 35, 73, 58, 60, 68, 73, 4 };
                int[] targets = new int[25] { 69, 58, 4, 66, 68, 73, 69, 62, 69, 60, 13, 4, 76, 63, 58, 69, 4, 35, 73, 58, 60, 68, 73, 4, 47 };
                Utility utility = new Utility();
                var hiddenPrevious = utility.RandomZeros(hiddenNeuron, 1);
                loss = LossCalcuation(inputs, targets, hiddenPrevious);
                smoothLoss = smoothLoss * 0.999 + loss * 0.001;
                if (n % 1000 == 0)
                {
                    PrintText(hiddenPrevious, inputs[0], 200);
                }
                n++;
            }

        }

        double LossCalcuation(int[] inputs, int[] targets, double[,] previousHidden)
        {
            double loss = 0;
            //if (File.Exists(textFile))
            //{
            Utility utility = new Utility();
            // Read entire text file content in one string  
            //string text = File.ReadAllText(textFile);
            //var charList = text.ToCharArray();
            //var uniqueChars = charList.Distinct();
            //int vocab_size = uniqueChars.Count();
            //double[,] Wxh = utility.RandomValues(hiddenNeuron, vocab_size);
            //double[,] Whh = utility.RandomValues(hiddenNeuron, hiddenNeuron);
            //double[,] Why = utility.RandomValues(vocab_size, hiddenNeuron);

            //double[,] bh = utility.RandomZeros(hiddenNeuron, 1);
            //double[,] by = utility.RandomZeros(vocab_size, 1);


            var hiddenPrevious = utility.RandomZeros(hiddenNeuron, 1);

            Dictionary<int, double[,]> xs = new Dictionary<int, double[,]>();
            Dictionary<int, double[,]> hs = new Dictionary<int, double[,]>();
            Dictionary<int, double[,]> ys = new Dictionary<int, double[,]>();
            Dictionary<int, double[,]> propablity = new Dictionary<int, double[,]>();

            //feed forward
            hs[-1] = hiddenPrevious;
            for (int i = 0; i < inputs.Length; i++)
            {
                try
                {
                    xs[i] = (utility.RandomZeros(vocab_size, 1));
                    xs[i][inputs[i], 0] = 1;
                    hs[i] = utility.Tanh(utility.Add(utility.Add(utility.MatrixMutiply(Wxh, xs[i]), utility.MatrixMutiply(Whh, hs[i - 1])), bh));
                    //ys[i] = utility.MatrixMutiply(Why, hs[i]) + by;
                    ys[i] = utility.Add(utility.MatrixMutiply(Why, hs[i]), by);
                    propablity[i] = utility.Softmax(ys[i]);
                    loss = loss - Math.Log(propablity[i][targets[i], 0]);
                    //bs[i] = utility.Softmax(ys[i]);
                }
                catch (Exception ex)
                {

                }


            }

            double[,] dWxh = utility.RandomZeros(hiddenNeuron, vocab_size); ;
            double[,] dWhh = utility.RandomZeros(hiddenNeuron, hiddenNeuron); ;
            double[,] dWhy = utility.RandomZeros(vocab_size, hiddenNeuron); ;
            double[,] dbh = utility.RandomZeros(hiddenNeuron, 1);
            double[,] dby = utility.RandomZeros(vocab_size, 1);
            var dhnext = utility.RandomZeros(hs[0].Length, 1);

            //back propagate
            for (int j = inputs.Length - 1; j > 0; j--)
            {
                try
                {
                    var dY = propablity[j];
                    dY[targets[j], 0] -= 1;
                    dWhy = utility.Add(utility.MatrixMutiply(dY, utility.TransposeValues(hs[j])), dWhy);
                    dby = utility.Add(dby, by);
                    var dh = utility.Add(utility.MatrixMutiply(utility.TransposeValues(Why), dY), dhnext);
                    var dhraw = utility.Mutiply(utility.Substract(1, utility.Mutiply(hs[j], hs[j])), dh);
                    dbh = utility.Add(dhraw, dbh);
                    dWxh = utility.Add(utility.MatrixMutiply(dhraw, utility.TransposeValues(xs[j])), dWxh);
                    dWhh = utility.Add(utility.MatrixMutiply(dhraw, utility.TransposeValues(hs[j - 1])), dWhh);
                    dhnext = utility.MatrixMutiply(utility.TransposeValues(Whh), dhraw);
                }
                catch (Exception ex)
                {

                }


            }
            // }
            return loss;
        }


        void PrintText(double[,] previousHidden, int inputValue, int iteration)
        {

            Utility utility = new Utility();
            List<double[,]> axisList = new List<double[,]>();
            StringBuilder @string = new StringBuilder();
            try
            {
                var x = utility.RandomZeros(80, 1);
                //# customize it for our seed char
                //x[seed_ix] = 1
                x[inputValue, inputValue] = 1;//need to correct it  
                for (int i = 0; i < iteration; i++)
                {
                    previousHidden = utility.Tanh(utility.Add(utility.Add(utility.MatrixMutiply(Wxh, x), utility.MatrixMutiply(Whh, previousHidden)), bh));
                    var y = utility.Add(utility.MatrixMutiply(Why, previousHidden), by);
                    var p = utility.Softmax(y);
                    //highest values has to bind.
                    //# pick one with the highest probability 
                    //ix = np.random.choice(range(vocab_size), p = p.ravel())
                    var ix = utility.NextInteger(1, vocab_size);
                    x[ix, i] = 1;
                    axisList.Add(x);
                }

                foreach (var y in axisList)
                {
                    // @string.Append(intToText[y])
                }
            }
            catch (Exception ex)
            {

            }



            Console.WriteLine("Print text for the output");
        }
    }
}
