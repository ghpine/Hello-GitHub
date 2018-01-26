from mlib import number


def P(n, k):
	''' P(n, k) = 2^{n-k}((k-1)2^n + n - k + 1) / (2^n - 1)^2'''
	nk2 = 2 ** (n - k)
	# n2 = nk2 * 2 ** k
	n2 = 2 ** n
	num = nk2 * ((k - 1) * n2 + n - k + 1)
	den = (n2 - 1) * (n2 - 1)
	g = number.gcd(num, den)
	# print g,
	return num / g, den / g


def test1():
	# n, d = P(100000000+7,10000+7)
	for i in range(3, 30):
		for j in range(1, i + 1):
			n, d = P(i, j)
			print n,
		print


def sol1():
	n = 100000000 + 7
	k = 10000 + 7
	m = 1000000000
	x = 1
	for i in xrange(n):
		x <<= 1
		x %= m
	y = 1
	for i in xrange(n - k):
		y <<= 1
		y %= m
	num = y * ((k - 1) * x + n - k + 1)
	den = (x - 1) * (x - 1)
	print num * den


# returns 2^n modulo mod
def powerOf2(n, mod):
	x = 2
	value = 1
	while (n > 0):
		if (n & 1) == 1:
			value = (value * x) % mod
		x = (x * x) % mod
		n >>= 1
	return value


# returns gcd(2^n-1, n) for large n
def gcd(n):
	return number.gcd(powerOf2(n, n) - 1, n)


def sol2():
	n = int(1e8 + 7)
	k = int(1e4 + 7)
	print "gcd(2^{0} - 1, {1}) = {2}".format(n, n, gcd(100000000 + 7))

	mod = int(1e9)
	p2n = powerOf2(n, mod)
	num = powerOf2(n - k, mod) * ((k - 1) * (p2n - 1) + n) % mod
	den = (p2n - 1) * (p2n - 1) % mod
	print num * den % mod


sol2()
