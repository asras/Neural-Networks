// Tests.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace Eigen;

int main()
{
	string f = "------------";
	MatrixXd m(2, 2);
	m(0, 0) = 0;
	m(1, 0) = 1;
	m(0, 1) = 1;
	m(1, 1) = 1;
	cout << m << endl;
	cout << f << endl;

	cout << "A random 3 by 3 matrix with values from -1 to 1." << endl;

	MatrixXd m2 = MatrixXd::Random(3, 3);

	cout << m2 << endl;
	cout << f << endl;

	cout << "A zero matrix." << endl;
	MatrixXd A = MatrixXd::Zero(2, 2);
	cout << A << endl;
	cout << f << endl;


	cout << "A matrix multiplication." << endl;
	MatrixXd s = MatrixXd::Zero(2, 2);
	s(0, 0) = 1;
	s(1, 1) = -1;
	cout << s << endl;
	cout << "Multiplied by" << endl;


	MatrixXd d = MatrixXd::Zero(2, 2);
	d(0, 1) = 1;
	d(1, 0) = 1;
	cout << d << endl << "equals" << endl;
	cout << s*d << endl;
	cout << f << endl;

	cout << "Vector constructor with argument." << endl;
	VectorXf * tester = new VectorXf(2);
	cout << *tester << endl;
	delete tester;
	cout << f << endl;


	cout << "What does the constructor do man" << endl;
	VectorXf yo2(2);
	cout << "yo2: " << endl << yo2 << endl;
	VectorXf yo3(3);
	cout << "yo3: " << endl << yo3 << endl;
	cout << f << endl;

	cout << "Array of vectors" << endl;
	VectorXf * pVecs = new VectorXf[3]; //I dont like this because it seems unlikely that it has a good estimate of how much memory
	//to allot each element in the array 
	for (int i = 0; i < 3; i++) {
		pVecs[i] = *new VectorXf(i + 1); //this is so weird
	}
	for (int i = 0; i < 3; i++) {
		cout << pVecs[i] << endl;
		cout << ":) (:" << endl;
	}
	delete[] pVecs;
	cout << f << endl;


	cout << "Multiplying a matrix by a number." << endl;
	MatrixXf mf = MatrixXf::Zero(2,2);
	mf(0, 0) = 0;
	mf(0, 1) = 1;
	mf(1, 0) = 2;
	mf(1, 1) = 3;

	cout << mf * 3 << endl;
	cout << f << endl;

	cout << "Inline multiplication" << endl;
	mf = MatrixXf::Random(2, 2)*1000;
	cout << mf << endl;
	cout << f << endl;


	cout << "Selecting a specific column" << endl;
	mf = MatrixXf::Zero(2, 5);
	mf(0, 0) = 1;
	mf(1, 0) = 1;
	mf(0, 1) = 2;
	mf(1, 1) = 2;
	mf(0, 2) = 3;
	mf(1, 2) = 3;
	mf(0, 3) = 4;
	mf(1, 3) = 4;
	mf(0, 4) = 5;
	mf(1, 4) = 5;
	cout << "The first column:" << endl << mf.col(0) << endl;
	cout << "The third column:" << endl << mf.col(2) << endl;
	cout << f << endl;

	cout << "Element-wise operations." << endl;
	VectorXf elVec(3);
	elVec(0) = 1;
	elVec(1) = 2;
	elVec(2) = 3;
	VectorXf tanhElVec = elVec.array().tanh();
	VectorXf invElVec = elVec.array().inverse();
	cout << "El-wise tanh" << endl << tanhElVec << endl;
	cout << "El-wise inverse" << endl << invElVec << endl;
	cout << "El-wise + 1" << endl << elVec.array() + 1 << endl;
	VectorXf softmax = ((-elVec.array()).exp() + 1).inverse();
	cout << "Softmax:" << endl << softmax << endl;
	cout << f << endl;


	cout << "Get number of rows and columns." << endl;
	MatrixXf mvp = MatrixXf::Random(2, 3);
	cout << "Rows: " << mvp.rows() << endl << "Cols: " << mvp.cols() << endl;
	cout << f << endl;


	std::cin.get();
    return 0;
}

