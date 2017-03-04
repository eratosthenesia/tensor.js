// Take the absolute value of a number
function abs(x) {
  return x < 0 ? -x : x;
}

// range(0,4,2) = 0,2,4
function range(a, b, j) {
  j = j === undefined ? 1 : j;
  var r = new Array(Math.floor(abs(b - a) / j));
  var i = 0;
  if (j > 0) {
    while (a <= b) {
      r[i] = a;
      a += j;
      i++;
    }
  } else if (j < 0) {
    while (a >= b) {
      r[i] = a;
      a -= j;
      i--;
    }
  }
  return r;
}

// Makes an array of n qs
function repeatit(n, q) {
  var r = new Array(n);
  for (var i = 0; i < n; ++i) {
    r[i] = q;
  }
  return r;
}

// Makes an array of n zeros
function zeros(n) {
  return repeatit(n, 0);
}

function flatten(xs) {
  if (typeof xs != 'object') {
    return xs;
  }
  for (var i = 0; i < xs.length; ++i) {
    xs[i] = flatten(xs[i]);
  }
  var r = [];
  for (var i = 0; i < xs.length; ++i) {
    r = r.concat(xs[i]);
  }
  return r;
}

function permutationParity(ns) {
  var vs = new Array(ns.length);
  var nv = 0;
  i = 0;
  m = 0;
  while (nv < ns.length) {
    j = i;
    i = ns[j];
    n = 0;
    vs[j] = true;
    nv++;
    while (i != j) {
      n++;
      vs[i] = true;
      i = ns[i];
      nv++;
    }
    m += n;
    i = 0;
    while ((vs[i]) && (i < ns.length)) {
      i++;
    }
  }
  return m % 2 == 0;
}

function factorial(n) {
  m = 1;
  while (n >= 2) {
    m *= n;
    n -= 1;
  }
  return m;
}

// Returns the nth lexographic permutation of xs
function permn(xs, n) {
  len = xs.length;
  f = factorial(len - 1);
  ps = [];
  m = len - 1;
  i = 0;
  while (n > 0) {
    idx = Math.floor(n / f);
    n %= f;
    f /= m;
    m -= 1;
    ps.push(xs[idx]);
    xs.splice(idx, 1);
    i++;
  }
  while (i < len) {
    i++;
    ps.push(xs[0]);
    xs.splice(0, 1);
  }
  return ps;
}

// Returns x mapped with func. Not sure why I wrote this.
function mapper(func) {
  return function(x) {
    return x.map(func);
  };
}

// Necessary for identifying operators
var sortedOps = ['+', '-', '*', ':='].sort();

// Takes a tensor expression, some tensors, etc. and makes a tensor out of it
function tensorFromTensorExpression(tensors, expr, tensorNamesOrder, n, context) {
  var tensorExpression = prepareTensorExpression(expr, tensorNamesOrder, n);
  var RPN = shuntingYardToRPN(tensorExpression.slice(1));
  var tensorFunc = funcRPNToFunction(RPN);
  var dataFunc = tensorExpression[0].func;
  var specs = tensorExpression[0].specs;
  var data = zeros(Math.pow(n, specs.rankup + specs.rankdown));
  var indices = zeros(specs.nindices);
  do {
    dataFunc(data, indices, tensorFunc(tensors, indices, context));
  } while (!baseincr(indices, n))
  return new Tensor(data, specs.rankup, specs.rankdown, n, undefined, specs.name);
}


function getTensorSpecs(tensorstring) {
  var getlower = false;
  var getupper = false;
  var getname = true;
  var upper = '';
  var lower = '';
  var name = '';
  for (var i = 0; i < tensorstring.length; ++i) {
    if (tensorstring[i] == '^') {
      upper = '^';
      getupper = true;
      getlower = false;
      getname = false;
    } else if (tensorstring[i] == '_') {
      lower = '_';
      getupper = false;
      getlower = true;
      getname = false;
    } else {
      if (getupper) {
        upper += tensorstring[i];
      } else if (getlower) {
        lower += tensorstring[i];
      } else if (getname) {
        name += tensorstring[i];
      }
    }
  }
  var getrank = function(name) {
    return name.length === 0 ? 0 : name.length - 1;
  }
  return {
    name: name,
    tensorFormatted: name + upper + lower,
    rankup: getrank(upper),
    rankdown: getrank(lower)
  };
}

function prepareTensorExpression(expr, tensorNamesOrder, n) {
  expr = expr.split(' ').join('');
  expr = expr.split(':=');
  if (expr.length != 2) {
    expr = expr.join('').split('=:');
    if (expr.length != 2) {
      return TensorError('Must exactly one := or =:');
    }
    expr = [expr[1], expr[0]];
  }
  expr = expr.join(':=');
  expr = splitString(expr, '+', '*', '-', ':=');
  expr.splice(1, 1);
  tensorNamesOrder = numberList(tensorNamesOrder).sort(function(x, y) {
    return x[0] > y[0];
  });
  var tensorNames = [];
  var indices = [];
  expr = expr.map(function(x) {
    if (!inSorted(x, sortedOps)) {
      if (!isNaN(x)) {
        return parseFloat(x);
      }
      specs = getTensorSpecs(x);
      x = splitStringNoKeep(specs.tensorFormatted, '^', '_');
      tensorNames.push(x[0]);
      indices.push(x.slice(1).join(''));
      return {
        name: x[0],
        indices: x.slice(1).join('').split(''),
        specs: specs
      };
    } else {
      return x;
    }
  });
  indices = weedUnique(indices.join('').split('').sort());
  indexIndexerer = mapper(binindexer(indices));
  tensorIndexer = binindexer(tensorNamesOrder, function(x, y) {
    return x[0] > y;
  }, function(x, y) {
    return x[0] == y;
  })
  expr = expr.map(function(x) {
    if (typeof x === 'object') {
      x.indexPicker = makePicker(indexIndexerer(x.indices));
      x.tensorIndex = tensorIndexer(x.name);
      delete x.indices;
      delete x.name;
      return x;
    } else {
      return x;
    }
  });
  var indexPicker = expr[0].indexPicker;
  var tensorIndex;
  expr[0] = (function(indexPicker) {
    return {
      specs: expr[0].specs,
      func: function(data, indices, exprValue) {
        data[baserep(indexPicker(indices), n)] += exprValue;
      }
    }
  })(indexPicker);
  for (var i = 1; i < expr.length; ++i) {
    if (typeof expr[i] !== 'number' && !inSorted(expr[i], sortedOps)) {
      indexPicker = expr[i].indexPicker;
      tensorIndex = expr[i].tensorIndex;
      expr[i] = (function(indexPicker, tensorIndex) {
        return function(tensors, indices, context) {
          return tensors[tensorIndex].getElement(indexPicker(indices));
        }
      })(expr[i].indexPicker, expr[i].tensorIndex);
    }
  }
  expr[0].specs.nindices = indices.length;
  return expr;
}

function argArgOp(arg1, arg2, op) {
  if (typeof arg1 === typeof arg2 === 'number') {
    return op(arg1, arg2);
  }
  argGetter = function(arg) {
    if (typeof arg === 'string') {
      return getNthArger(parseInt(arg));
    } else if (typeof arg === 'number') {
      return valueFunc(arg);
    } else {
      return arg;
    }
  }
  arg1 = argGetter(arg1);
  arg2 = argGetter(arg2);
  return function() {
    return op(arg1.apply(this, arguments), arg2.apply(this, arguments));
  }
}

function mapBottom(xss, func, atBottom) {
  atBottom = atBottom === undefined ? function(x) {
    return typeof x !== 'object';
  } : atBottom;
  var ret = new Array(xss.length);
  for (var i = 0; i < xss.length; ++i) {
    if (atBottom(xss[i])) {
      ret[i] = func(xss[i]);
    } else {
      ret[i] = mapBottom(xss[i], func, atBottom);
    }
  }
  return ret;
}

function joinFancy(xs, joiners) {
  if (joiners.length == 0) {
    return xs;
  }
  return intersplice(xs.map(function(x) {
    return joinFancy(x, joiners.slice(1));
  }), joiners[0]);
  //  joinFlatter(xs.map(function(x){return joinFancy(x,joiners.slice(1));}),joiners[0]);
}

function splitStringHelper(str, joiners) {
  if (typeof str === 'object') {
    return intersplice(str.map(function(x) {
      return joinFancy(x, joiners.slice(1));
    }), joiners[0]);
  }
  return str;
}

function splitString(str) {
  var splits = typeof arguments[1] === 'string' ? [].slice.call(arguments, 1) : arguments[1];
  var r = [str];
  for (var i = 0; i < splits.length; ++i) {
    r = mapBottom(r, function(s) {
      return s.split(splits[i]);
    });
  }
  return flatten(joinFancy(r[0], splits));
}

function splitStringNoKeep(str) {
  var splits = typeof arguments[1] === 'string' ? [].slice.call(arguments, 1) : arguments[1];
  var r = [str];
  for (var i = 0; i < splits.length; ++i) {
    r = mapBottom(r, function(s) {
      return s.split(splits[i]);
    });
  }
  return flatten(r);
}

function intersplice(xs, y) {
  return joinFlatter(xs.map(function(x) {
    return [x];
  }), y);
}

function joinFlatter(xss, glue) {
  var xs = new Array(xss.length - 1 + xss.map(function(x) {
    return x.length;
  }).reduce(plusOp));
  var j = 0;
  for (var i = 0; i < xss.length; ++i) {
    for (var k = 0; k < xss[i].length; ++k) {
      xs[j] = xss[i][k];
      ++j;
    }
    if (i != xss.length - 1) {
      xs[j] = glue;
    }
    ++j;
  }
  return xs;
}

function inSorted(n, h) {
  return binsearchindex(n, h) !== false;
}

function getNthArger(x) {
  return function() {
    return arguments[x];
  };
}

function valueFunc(x) {
  return function() {
    return x;
  }
}

function idFunc(x) {
  return x;
}

function minusOp(a, b) {
  return b - a;
}

function plusOp(a, b) {
  return b + a;
}

function timesOp(a, b) {
  return b * a;
}

function shuntingYardToRPN(opers) {
  var i = 0;
  var operstack = [];
  var numstack = [];
  var rpn = [];
  while (i < opers.length) {
    if ((typeof opers[i] === 'string' && opers[i] != '*' && opers[i] != '-' && opers[i] != '+') || typeof opers[i] === 'number' || typeof opers[i] === 'function') {
      numstack.push(opers[i]);
    } else if (opers[i] == '*') {
      operstack.push('*');
    } else if (opers[i] == '+' || opers[i] == '-') {
      while (operstack[operstack.length - 1] == '-' || operstack[operstack.length - 1] == '*') {
        rpn.push(numstack.pop());
        rpn.push(numstack.pop());
        rpn.push(operstack.pop());
      }
      operstack.push(opers[i]);
    }
    ++i;
  }
  while (num = numstack.pop()) {
    rpn.push(num);
  }
  while (oper = operstack.pop()) {
    rpn.push(oper);
  }
  return rpn;
}

function zipLists() {
  var lists = [].slice.call(arguments);
  var n = Math.min.apply(this, lists.map(function(x) {
    return x.length;
  }));
  var r = new Array(n);
  for (var i = 0; i < n; ++i) {
    r[i] = lists.map(function(x) {
      return x[i];
    });
  }
  return r;
}

function numberList(list) {
  return zipLists(list, range(0, list.length - 1));
}

function indicesWhereTrue(list, test) {
  var r = [];
  for (var i = 0; i < list.length; ++i) {
    if (test(list[i])) {
      r.push(i);
    }
  }
  return r;
}

function mapAtIndices(list, indicesSorted, map) {
  var r = new Array(list);
  var j = 0;
  var i = 0;
  while (i < list.length && j < indicesSorted.length) {
    for (; i < indicesSorted[j]; ++i) {
      r[i] = list[i];
    }
    r[i++] = map(list[indicesSorted[j++]]);
  }
  for (; i < list.length; ++i) {
    r[i] = list[i];
  }
  return r;
}

function binfind(n, h, a, b) {
  var idx = binsearchindex(n, h, a, b);
  if (idx !== false) {
    return h[idx];
  }
}

function simpleRPNToFunction(rpn, variables) {
  var i = 0;
  var stack = [];
  var variableIndices = numberList(variables).sort(function(x, y) {
    return x[0] > y[0];
  });
  var opDict = {
    '-': minusOp,
    '+': plusOp,
    '*': timesOp
  };
  var getVariableIdx = function(v) {
    return binfind(v, variableIndices, function(h, n) {
      return h[0] > n;
    }, function(h, n) {
      return h[0] == n;
    })[1]
  };
  for (; i < rpn.length; ++i) {
    if (inSorted(rpn[i], sortedOps)) {
      a = stack.pop();
      b = stack.pop();
      stack.push(argArgOp(a, b, opDict[rpn[i]]));
    } else if (typeof rpn[i] === 'number') {
      stack.push(rpn[i]);
    } else {
      stack.push(getVariableIdx(rpn[i]) + '');
    }
  }
  if (typeof stack[0] === 'number') {
    return valueFunc(stack[0]);
  } else if (typeof stack[0] === 'string') {
    return function() {
      return arguments[getVariableIdx(stack[0])];
    }
  }
  return stack[0];
}

function funcRPNToFunction(funcRPN) {
  var functionIndices = indicesWhereTrue(funcRPN, function(x) {
    return typeof x === 'function';
  });
  var functions = makePicker(functionIndices)(funcRPN);
  var i = 0;
  var rpnModified = mapAtIndices(funcRPN, functionIndices, function(func) {
    return (i++) + '';
  });
  var mainFunction = simpleRPNToFunction(rpnModified, range(0, i - 1).map(function(x) {
    return x + '';
  }));
  return function() {
    var args = [].slice.call(arguments);
    return mainFunction.apply(this, functions.map(function(func) {
      return func.apply(this, args);
    }));
  };
}

function leviCivita(ns) {
  ms = new Array(ns.length);
  for (var i = 0; i < ns.length; ++i) {
    ms[i] = ns[i];
  }
  ms.sort();
  for (var i = 0; i < ns.length; ++i) {
    if (ms[i] != i) {
      return 0;
    }
  }
  return permutationParity(ns) ? 1 : -1;
}

function tensorLookup(data) {
  if (arguments.length > 2) {
    indices = [].slice.call(arguments, 1);
  } else {
    if (typeof arguments[1] == "object") {
      indices = arguments[1];
    } else {
      indices = [arguments[1]];
    }
  }
  if (data.hasOwnProperty('data')) {
    var context = data.context;
    data = data.data;
  }
  var i = 0;
  while (i < indices.length) {
    data = data[indices[i]];
    ++i;
  }
  if (typeof data === 'function') {
    data = data(context);
  }
  return data;
}

function JCBError(message) {
  this.message = message;
  this.talk = function() {
    console.log(this.message);
  };
}

function getProperties(obj) {
  var r = [];
  for (var k in obj) {
    if (obj.hasOwnProperty(k)) {
      r = r.concat(k);
    }
  }
  return r;
}

var inherit = function(parent, child, traits, override) {
  if (traits === undefined) {
    traits = getProperties(parent);
  }
  if (override === undefined) {
    override = this.defaults.override;
  }
  for (var i = 0; i < traits.length; ++i) {
    if (!child.hasOwnProperty(traits[i]) || override) {
      child[traits[i]] = parent[traits[i]];
    }
  }
}

inherit = inherit.bind({
  defaults: {
    override: true
  }
});

function TensorError(message) {
  var parent = JCBError(message);
  inherit(parent, this);
  this.errorType = "Tensor";
}

function errorClass(errorType) {
  var clase = function(message) {
    var parent = JCBError(message);
    inherit(parent, this);
    this.errorType = errorType;
  };
  return clase;
}

function fxTensorTimes(other, op) {
  var proddata = {
    first: {
      data: this.data,
      getElementFunction: this.getElementFunction
    },
    second: {
      data: other.data,
      getElementFunction: other.getElementFunction
    }
  }
  var thisrankup = (this.index >= this.rankup ? 0 : -1) + this.rankup;
  var otherrankup = (other.index >= other.rankup ? 0 : -1) + other.rankup;
  var thisrankup = (this.index >= this.rankup ? -1 : 0) + this.rankup;
  var otherrankup = (other.index >= other.rankup ? -1 : 0) + other.rankup;
  var prodrankup = thisrankup + otherrankup;
  var prodrankdown = thisrankdown + otherrankdown;
  var self = this;
  var prodCustomTensorLookup = function(data) {
    if (arguments.length > 2) {
      indices = [].slice.call(arguments);
    } else {
      if (typeof arguments[1] == "object") {
        indices = arguments[1];
      } else {
        indices = [arguments[1]];
      }
    }
    return op(data.first.getElementFunction(data.first.data,
        indices.slice(0, thisrankup).concat(
          indices.slice(thisrankup + otherrankup, thisrankup + otherrankup + thisrankdown))),
      data.second.getElementFunction(data.decond.data,
        indices.slice(thisrankup, thisrankup + otherrankup).concat(
          indices.slice(thisrankup + otherrankup + thisrankdown, prodrankup + prodrankdown))));
  }
  return Tensor(proddata, prodrankup, prodrankdown, this.n, prodCustomTensorLookup);
}
// Before we do tensorTimes, we must define
// this.dummyIndex = n, where n is the index
// we're summing over. This is the simplest
// equation, and we use it to build more complex
// ones.

function baseincr(ns, b) {
  var i = 0;
  ns[i]++;
  carry = false;
  while (i < ns.length && ns[i] == b) {
    ns[i] = 0;
    if (i < ns.length - 1) {
      ns[++i]++;
    } else {
      carry = true;
    }
  }
  return carry;
}

function baserep(ns, b) {
  var i = ns.length - 1;
  var n = 0;
  while (i >= 0) {
    n = n * b + ns[i--];
  }
  return n;
}

function tensorTimes(self, selfidx, them, themidx, op) {
  if (op == undefined) {
    op = function(a, b) {
      return a * b;
    };
  }
  var n = self.n;
  var rankup = (selfidx >= self.rankup ? -1 : 0) +
    (themidx >= them.rankup ? -1 : 0) +
    self.rankup + them.rankup;
  var rankdown = (selfidx >= self.rankup ? 0 : -1) +
    (themidx >= them.rankup ? 0 : -1) +
    self.rankdown + them.rankdown;
  var nindices = rankup + rankdown;
  var indices = zeros(nindices);

  function indicesincr() {
    baseincr(indices, n);
  }

  function getDummyIndexed(dummy, a, b, c) {
    r = indices.slice(a, a + c);
    r.splice(b, 0, dummy);
    return r;
  }
  var nsums = Math.pow(n, nindices);
  var sums = zeros(nsums);
  for (var k = 0; k < nsums; ++k) {
    for (var dummy = 0; dummy < n; ++dummy) {
      selfIndices = getDummyIndexed(dummy, 0, selfidx, self.rankup + self.rankdown - 1);
      themIndices = getDummyIndexed(dummy, self.rankup + self.rankdown - 1, themidx, them.rankup + them.rankdown - 1);
      sums[k] += op(self.getElement(selfIndices), them.getElement(themIndices));
    }
    indicesincr();
  }
  return new Tensor(sums, rankup, rankdown, n);
}

function compactTensorData(linearData, n, reclevel) {
  var m = n;
  while (m < linearData.length) {
    m *= n;
  }
  m /= n;
  var r = new Array(n);
  if (m == 1) {
    r = linearData;
  } else {
    var i = 0;
    for (var j = 0; j < linearData.length; j += m) {
      r[i] = compactTensorData(linearData.slice(j, j + m), n, reclevel + 1);
      ++i;
    }
  }
  return r;
}

function leviCivitaTensor(numIndices) {
  if (numIndices === undefined) {
    numIndices = 3;
  }
  var lc = new Tensor([], 0, numIndices, -1, function(_, indices) {
    return leviCivita(indices);
  });
  return lc;
}

function kroneckerDeltaTensor() {
  var d = new Tensor([], 1, 1, -1, function(_, indices) {
    return indices[0] == indices[1] ? 1 : 0;
  });
  return d;
}

function logab(a, b) {
  r = 0;
  s = 1;
  while (s <= b) {
    s *= a;
    r++;
  }
  r--;
  return r;
}

function weedUnique(sortedList, neq) {
  if (neq === undefined) {
    neq = function(a, b) {
      return a != b;
    };
  }
  var uniques = [sortedList[0]];
  var last = sortedList[0];
  var i = 1;
  while (i < sortedList.length) {
    if (neq(sortedList[i], last)) {
      last = sortedList[i];
      uniques.push(sortedList[i]);
    }
    ++i;
  }
  return uniques;
}

function minusOrdset(Aset, Bset, gt, eql) {
  if (gt === undefined) {
    gt = function(a, b) {
      return a > b;
    };
  }
  if (eql === undefined) {
    eql = function(a, b) {
      return a == b;
    };
  }
  var Mset = [];
  var a = 0;
  var b = 0;
  while (a < Aset.length && b < Bset.length) {
    while (gt(Aset[a], Bset[b]) && b < Bset.length) {
      b++;
    }
    while (eql(Aset[a], Bset[b]) && a < Aset.length && b < Bset.length) {
      a++;
      b++;
    }
    Mset.push()
    while (gt(Bset[b], Aset[a]) && a < Aset.length) {
      Mset.push(Aset[a++]);
    }
  }
  while (a < Aset.length) {
    Mset.push(Aset[a++]);
  }
  return Mset;
}

function onePicker(pickIndex) {
  return function(list) {
    return list[pickIndex];
  }
}

function makePicker(pickIndices) {
  return function(list) {
    var r = new Array(pickIndices.length);
    for (var i = 0; i < pickIndices.length; ++i) {
      r[i] = list[pickIndices[i]];
    }
    return r;
  };
}

function binsearchindex(needle, haystack, gt, eql) {
  if (undefined === gt) {
    gt = function(a, b) {
      return a > b;
    };
  }
  if (undefined === eql) {
    eql = function(a, b) {
      return a == b;
    };
  }
  var a = 0;
  var b = haystack.length;
  var c;
  while (a != b) {
    c = (a + b) >> 1;
    if (eql(haystack[c], needle)) {
      return c;
    } else if (gt(haystack[c], needle)) {
      b = c;
    } else {
      a = c + 1;
    }
  }
  return false;
}

function collate(xs, ys) {
  var i = 0;
  var j = 0;
  var r = [];
  while (i < xs.length && j < ys.length) {
    r.push(xs[i++]);
    r.push(ys[j++]);
  }
  while (i < xs.length) {
    r.push(xs[i++]);
  }
  while (j < ys.length) {
    r.push(ys[j++]);
  }
  return r;
}

function simpleShuntingYard(opers, variables) {
  var i = 0;
  var operstack = [];
  var numstack = [];
  variables = variables === undefined ? [] : variables;
  var sortedVariables = variables.slice().sort();
  var rpn = [];
  while (i < opers.length) {
    if (typeof opers[i] === 'string' || typeof opers[i] === 'number') {
      numstack.push(opers[i]);
    } else if (opers[i] == '*') {
      operstack.push('*');
    } else if (opers[i] == '+' || opers[i] == '-') {
      while (operstack[operstack.length - 1] == '-' || operstack[operstack.length - 1] == '*') {
        rpn.push(numstack.pop());
        rpn.push(numstack.pop());
        rpn.push(operstack.pop());
      }
      operstack.push(opers[i]);
    }
    ++i;
  }
  while (oper = operstack.pop()) {
    rpn.push(oper);
  }
  return rpn;
}

function binindexer(haystack, gt, eql) {
  return function(needle) {
    return binsearchindex(needle, haystack, gt, eql);
  }
}

function TensorChain(tensor, name, chain) {
  if (chain === undefined) {
    this.chain = [];
  } else {
    this.chain = chain;
  }
  if (name !== undefined) {
    tensor.name = name;
  }
  this.chain.push(tensor);
  this.n = tensor.n;
  this.m = function(tensor, name) {
    return new TensorChain(tensor, name, this.chain)
  }
  this.eq = function(expr, n, context) {
    if (n !== undefined) {
      this.n = n;
    }
    var tensorNamesOrder = this.chain.map(function(x) {
      return x.name;
    });
    return tensorFromTensorExpression(this.chain, expr, tensorNamesOrder, this.n, context);
  }
}

// The main tensor class.
function Tensor(data, rankup, rankdown, n, customTensorLookup, name) {
  if (data.hasOwnProperty('data')) {
    rankup = data.rankup;
    rankdown = data.rankdown;
    n = data.n;
    customTensorLookup = data.customTensorLookup;
    name = data.name;
    data = data.data;
  }
  if (n === undefined) {
    n = data.length;
  }
  if (customTensorLookup == undefined) {
    this.getElementFunction = tensorLookup;
    this.funcTensor = false;
  } else {
    this.getElementFunction = customTensorLookup;
    this.funcTensor = true;
  }
  this.getElement = function() {
    var context = [];
    var indices;
    if (arguments.length > 1) {
      indices = [].slice.call(arguments);
      context = indices.slice(this.rankup + this.rankdown);
      indices = indices.slice(0, this.rankup + this.rankdown);
    } else {
      if (typeof arguments[0] == "object") {
        indices = arguments[0];
      } else {
        indices = [arguments[0]];
      }
    }
    return this.getElementFunction(context.length === 0 ? this.data : {
      data: this.data,
      context: this.context
    }, indices);
  }
  this.inspect = function() {
    var r = 'Tensor<' + this.rankup + ',' + this.rankdown + '>:' + this.n;
    return r;
  }
  var logndatalen = logab(n, flatten(data).length);
  this.data = n < data.length && n != -1 ? compactTensorData(flatten(data), n) : data;
  this.rankup = rankup;
  this.rankdown = (rankup + rankdown == logndatalen) || (n == -1) ? rankdown : logndatalen - rankup;
  this.staticCopy = function(nprime) {
    var n;
    if (nprime === undefined) {
      if (this.n == -1) {
        return TensorError("N must be specified.");
      } else {
        n = this.n;
      }
    } else {
      if (this.n == -1) {
        n = nprime;
      } else {
        n = this.n;
      }
    }
    var nindices = this.rankup + this.rankdown;
    var indices = zeros(nindices);
    var data = new Array(Math.pow(n, nindices));
    var k = 0;
    do {
      data[k++] = this.getElement(indices);
    } while (!baseincr(indices, n));
    return new Tensor(data, this.rankup, this.rankdown, n);
  }
  this.times = function(idx, them, themidx, op) {
    if (them)
      return tensorTimes(this, idx, them, themidx, op);
  };
  this.m = function(name) {
    return new TensorChain(this, name);
  }
  this.fxTimes = fxTensorTimes;
  this.n = n;
}