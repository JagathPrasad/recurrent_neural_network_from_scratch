using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecurrentNeuralNetwork
{
    public class Utility
    {

        #region Calculation


        public int NextInteger(int min,int max)
        {
            Random random = new Random();
            return random.Next(min, max);
        }

        public double[,] Add(double[,] x, double[,] y)
        {
            int a = x.GetLength(0);
            int b = x.GetLength(1);
            double[,] z = new double[a, b];
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    try
                    {
                        z[i, j] = x[i, j] + y[i, j];
                    }
                    catch (Exception ex)
                    {

                    }

                }
            }
            return z;
        }


        public double[,] Subtract(double[,] x, double[,] y)
        {
            var a = x.GetLength(0);
            var b = x.GetLength(1);
            double[,] z = new double[a, b];
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    try
                    {
                        z[i, j] = x[i, j] - y[i, j];
                    }
                    catch (Exception ex)
                    {

                    }
                }
            }
            return z;
        }


        public double[,] Substract(double x, double[,] y)
        {
            var a = y.GetLength(0);
            var b = y.GetLength(1);
            double[,] z = new double[a, b];
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    try
                    {
                        z[i, j] = x - y[i, j];
                    }
                    catch (Exception ex)
                    {

                    }
                }
            }
            return z;
        }

        public double[,] Mutiply(double[,] x, double[,] y)
        {
            int a = y.GetLength(0);
            int b = y.GetLength(1);
            double[,] z = new double[a, b];
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    z[i, j] = x[i, j] * y[i, j];
                }
            }
            return z;
        }

        public double[,] MatrixMutiply(double[,] x, double[,] y)
        {
            int a = y.GetLength(0);
            int b = y.GetLength(1);//need to understand
            double[,] z = new double[a, b];
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    try
                    {
                        z[i, j] = y[i, j] * x[j, i];

                    }
                    catch (Exception ex)
                    {

                    }

                }
            }
            return z;
        }
        public double[][] MatrixMutiply(double[][] x, double[][] y)
        {
            int a = y.Length;
            int b = y.Length;//need to understand
            double[][] z = new double[a][];
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    try
                    {
                        z[i][j] = x[i][j] * y[j][i];

                    }
                    catch (Exception ex)
                    {

                    }

                }
            }
            return z;
        }


        #endregion



        public double[,] Softmax(double[,] x)
        {
            //formula
            //e^x/Sum(e^x)

            int a = x.GetLength(0);
            int b = x.GetLength(1);
            double[,] y = new double[a, b];
            double sum = 0;
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    sum = sum + Math.Exp(x[i, j]);
                }
            }
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    y[i, j] = Math.Exp(x[i, j]) / sum;
                }
            }
            return y;
        }

        public double[] Softmax(double[] x)
        {
            //formula
            //e^x/Sum(e^x)

            int a = x.Length;
            //int b = x[0].Length;
            double[] y = new double[a];
            double sum = 0;
            for (int i = 0; i < a; i++)
            {
                //for (int j = 0; j < b; j++)
                {
                    sum = sum + Math.Exp(x[i]);
                }
            }
            for (int i = 0; i < a; i++)
            {
                //for (int j = 0; j < b; j++)
                {
                    y[i] = Math.Exp(x[i]) / sum;
                }
            }
            return y;
        }

        public double[,] Tanh(double[,] x)
        {
            var tanh = Sigmoid(x);
            int t = tanh.GetLength(0);
            var t2 = tanh.GetLength(1);
            double[,] output = new double[t, t2];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t2; j++)
                {
                    try
                    {
                        output[i, j] = (tanh[i, j] * 2) - 1;
                    }
                    catch (Exception ex)
                    {
                    }

                }

            }
            //tanh formulae
            //(2*Sigmoid(x)-1)
            //(e^x-e^-x)/(e^x+e^-x)
            return output;
        }

        public double[,] Sigmoid(double[,] x)
        {
            //formula
            //1/(1+e^-x)
            int a = x.GetLength(0);
            int b = x.GetLength(1);
            double[,] y = new double[a, b];

            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    try
                    {
                        y[i, j] = 1.0 / (1 + Math.Exp(-x[i, j]));
                    }
                    catch (Exception ex)
                    {

                    }
                }
            }
            return y;
        }

        public double Error()
        {
            return 0.0;
        }



        public Dictionary<string, int> ConvertTexttoInteger(char[] charList)
        {
            Dictionary<string, int> charToInteger = new Dictionary<string, int>();
            for (int i = 0; i < charList.Length; i++)
            {

                if (!charToInteger.ContainsKey(charList[i].ToString()))
                    charToInteger.Add(charList[i].ToString(), i);
            }
            return charToInteger;
        }

        public Dictionary<int, string> ConvertIntegertoText(char[] charList)
        {
            Dictionary<int, string> intToChar = new Dictionary<int, string>();
            for (int i = 0; i < charList.Length; i++)
            {
                if (!intToChar.ContainsKey(i))
                    intToChar.Add(i, charList[i].ToString());
            }
            return intToChar;
        }


        public double NextDouble()
        {
            Random random = new Random();
            return random.NextDouble();
        }


        public double[,] RandomValues(int x, int y)
        {
            double[,] randomValues = new double[x, y];
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    //randomValues[i][j] = 0.0 + (NextDouble() * (1.0 - 0.0));
                    randomValues[i, j] = (NextDouble() * 0.01);
                }
            }
            return randomValues;
        }

        public double[][] RandomVValues(int x, int y)
        {
            double[][] randomValues = new double[x][];
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    //randomValues[i][j] = 0.0 + (NextDouble() * (1.0 - 0.0));
                    randomValues[i][j] = (NextDouble() * 0.01);
                }
            }
            return randomValues;
        }


        public double[,] RandomZeros(int x, int y = 1)
        {
            double[,] zeros = new double[x, y];
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    zeros[i, j] = 0;
                }
            }
            return zeros;
        }
        public double[,] TransposeValues(double[,] x)
        {
            int a = x.GetLength(0);
            int b = x.GetLength(1);
            double[,] y = new double[b, a];
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    try
                    {
                        y[j, i] = x[i, j];
                    }
                    catch (Exception ex)
                    {

                        //throw;
                    }
                   
                }
            }
            return y;
        }

    }
}
