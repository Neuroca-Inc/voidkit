#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArithmeticError {
    Allocation,
    DivisionByZero,
    NonExactDivision,
    NegativeNatural,
    InvalidRational,
    SizeOverflow,
}

#[derive(Clone, Default, Eq, Hash, PartialEq)]
pub struct BigNat {
    limbs: Vec<u64>,
}

impl BigNat {
    #[must_use]
    pub const fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    pub fn try_from_u64(value: u64) -> Result<Self, ArithmeticError> {
        if value == 0 {
            return Ok(Self::zero());
        }
        let mut limbs = Vec::new();
        limbs
            .try_reserve_exact(1)
            .map_err(|_| ArithmeticError::Allocation)?;
        limbs.push(value);
        Ok(Self { limbs })
    }

    pub fn try_from_be_bytes(bytes: &[u8]) -> Result<Self, ArithmeticError> {
        let first = bytes.iter().position(|byte| *byte != 0).unwrap_or(bytes.len());
        if first == bytes.len() {
            return Ok(Self::zero());
        }
        let live = &bytes[first..];
        let limb_count = live
            .len()
            .checked_add(7)
            .ok_or(ArithmeticError::SizeOverflow)?
            / 8;
        let mut limbs = Vec::new();
        limbs
            .try_reserve_exact(limb_count)
            .map_err(|_| ArithmeticError::Allocation)?;
        for chunk in live.rchunks(8) {
            let mut limb = 0_u64;
            for byte in chunk {
                limb = (limb << 8) | u64::from(*byte);
            }
            limbs.push(limb);
        }
        let mut result = Self { limbs };
        result.normalize();
        Ok(result)
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    #[must_use]
    pub fn is_one(&self) -> bool {
        self.limbs.as_slice() == [1]
    }

    #[must_use]
    pub fn limb_len(&self) -> usize {
        self.limbs.len()
    }

    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.limbs.last().map_or(0, |last| {
            (self.limbs.len() - 1) * 64 + (64 - last.leading_zeros() as usize)
        })
    }

    #[must_use]
    pub fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    pub fn try_clone(&self) -> Result<Self, ArithmeticError> {
        let mut limbs = Vec::new();
        limbs
            .try_reserve_exact(self.limbs.len())
            .map_err(|_| ArithmeticError::Allocation)?;
        limbs.extend_from_slice(&self.limbs);
        Ok(Self { limbs })
    }

    pub fn try_add(&self, other: &Self) -> Result<Self, ArithmeticError> {
        let max_len = self.limbs.len().max(other.limbs.len());
        let capacity = max_len
            .checked_add(1)
            .ok_or(ArithmeticError::SizeOverflow)?;
        let mut limbs = Vec::new();
        limbs
            .try_reserve_exact(capacity)
            .map_err(|_| ArithmeticError::Allocation)?;

        let mut carry = 0_u128;
        for index in 0..max_len {
            let left = u128::from(*self.limbs.get(index).unwrap_or(&0));
            let right = u128::from(*other.limbs.get(index).unwrap_or(&0));
            let sum = left + right + carry;
            limbs.push(sum as u64);
            carry = sum >> 64;
        }
        if carry != 0 {
            limbs.push(carry as u64);
        }
        Ok(Self { limbs })
    }

    pub fn try_add_u64(&self, value: u64) -> Result<Self, ArithmeticError> {
        self.try_add(&Self::try_from_u64(value)?)
    }

    pub fn try_sub(&self, other: &Self) -> Result<Self, ArithmeticError> {
        if self < other {
            return Err(ArithmeticError::NegativeNatural);
        }
        let mut result = self.try_clone()?;
        result.sub_assign(other);
        Ok(result)
    }

    pub fn try_mul(&self, other: &Self) -> Result<Self, ArithmeticError> {
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero());
        }
        let output_len = self
            .limbs
            .len()
            .checked_add(other.limbs.len())
            .ok_or(ArithmeticError::SizeOverflow)?;
        let mut limbs = Vec::new();
        limbs
            .try_reserve_exact(output_len)
            .map_err(|_| ArithmeticError::Allocation)?;
        limbs.resize(output_len, 0);

        for (left_index, left) in self.limbs.iter().copied().enumerate() {
            let mut carry = 0_u128;
            for (right_index, right) in other.limbs.iter().copied().enumerate() {
                let index = left_index + right_index;
                let product = u128::from(left) * u128::from(right)
                    + u128::from(limbs[index])
                    + carry;
                limbs[index] = product as u64;
                carry = product >> 64;
            }
            limbs[left_index + other.limbs.len()] = carry as u64;
        }

        let mut result = Self { limbs };
        result.normalize();
        Ok(result)
    }

    pub fn try_shl_bits(&self, shift: usize) -> Result<Self, ArithmeticError> {
        if self.is_zero() {
            return Ok(Self::zero());
        }
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let extra = if bit_shift != 0 { 1 } else { 0 };
        let output_len = self
            .limbs
            .len()
            .checked_add(word_shift)
            .and_then(|value| value.checked_add(extra))
            .ok_or(ArithmeticError::SizeOverflow)?;
        let mut limbs = Vec::new();
        limbs
            .try_reserve_exact(output_len)
            .map_err(|_| ArithmeticError::Allocation)?;
        limbs.resize(output_len, 0);

        for (index, limb) in self.limbs.iter().copied().enumerate() {
            let target = index + word_shift;
            limbs[target] |= limb << bit_shift;
            if bit_shift != 0 {
                limbs[target + 1] |= limb >> (64 - bit_shift);
            }
        }
        let mut result = Self { limbs };
        result.normalize();
        Ok(result)
    }

    pub fn try_shr_bits(&self, shift: usize) -> Result<Self, ArithmeticError> {
        let mut result = self.try_clone()?;
        result.shr_assign(shift);
        Ok(result)
    }

    pub fn try_gcd(&self, other: &Self) -> Result<Self, ArithmeticError> {
        if self.is_zero() {
            return other.try_clone();
        }
        if other.is_zero() {
            return self.try_clone();
        }

        let common_shift = self.trailing_zeros().min(other.trailing_zeros());
        let mut left = self.try_shr_bits(self.trailing_zeros())?;
        let mut right = other.try_shr_bits(other.trailing_zeros())?;

        loop {
            match left.cmp(&right) {
                Ordering::Equal => return left.try_shl_bits(common_shift),
                Ordering::Greater => {
                    left.sub_assign(&right);
                    left.shr_assign(left.trailing_zeros());
                }
                Ordering::Less => {
                    right.sub_assign(&left);
                    right.shr_assign(right.trailing_zeros());
                }
            }
        }
    }

    pub fn try_div_rem(&self, divisor: &Self) -> Result<(Self, Self), ArithmeticError> {
        if divisor.is_zero() {
            return Err(ArithmeticError::DivisionByZero);
        }
        if self < divisor {
            return Ok((Self::zero(), self.try_clone()?));
        }

        let mut quotient_limbs = Vec::new();
        quotient_limbs
            .try_reserve_exact(self.limbs.len())
            .map_err(|_| ArithmeticError::Allocation)?;
        quotient_limbs.resize(self.limbs.len(), 0);
        let mut quotient = Self {
            limbs: quotient_limbs,
        };

        let remainder_capacity = divisor
            .limbs
            .len()
            .checked_add(1)
            .ok_or(ArithmeticError::SizeOverflow)?;
        let mut remainder_limbs = Vec::new();
        remainder_limbs
            .try_reserve_exact(remainder_capacity)
            .map_err(|_| ArithmeticError::Allocation)?;
        let mut remainder = Self {
            limbs: remainder_limbs,
        };

        for bit in (0..self.bit_len()).rev() {
            remainder.shl_one_reserved();
            if self.bit(bit) {
                remainder.add_one_reserved();
            }
            if &remainder >= divisor {
                remainder.sub_assign(divisor);
                quotient.set_bit_reserved(bit);
            }
        }

        quotient.normalize();
        remainder.normalize();
        Ok((quotient, remainder))
    }

    pub fn try_div_exact(&self, divisor: &Self) -> Result<Self, ArithmeticError> {
        let (quotient, remainder) = self.try_div_rem(divisor)?;
        if remainder.is_zero() {
            Ok(quotient)
        } else {
            Err(ArithmeticError::NonExactDivision)
        }
    }

    #[must_use]
    pub fn cmp_power_of_two(&self, exponent: &Self) -> Ordering {
        if self.is_zero() {
            return Ordering::Less;
        }
        let top_bit = self.bit_len() - 1;
        match exponent.cmp_usize(top_bit) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => {
                if self.is_power_of_two() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
        }
    }

    #[must_use]
    pub fn cmp_usize(&self, value: usize) -> Ordering {
        if self.limbs.len() > 1 {
            return Ordering::Greater;
        }
        self.limbs
            .first()
            .copied()
            .unwrap_or(0)
            .cmp(&(value as u64))
    }

    pub fn try_to_usize(&self) -> Result<usize, ArithmeticError> {
        if self.cmp_usize(usize::MAX) == Ordering::Greater {
            return Err(ArithmeticError::SizeOverflow);
        }
        Ok(self.limbs.first().copied().unwrap_or(0) as usize)
    }

    pub fn try_to_be_bytes(&self) -> Result<Vec<u8>, ArithmeticError> {
        if self.is_zero() {
            return Ok(Vec::new());
        }
        let byte_len = self.bit_len().div_ceil(8);
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| ArithmeticError::Allocation)?;
        bytes.resize(byte_len, 0);
        for (limb_index, limb) in self.limbs.iter().copied().enumerate() {
            for byte_index in 0..8 {
                let absolute = limb_index * 8 + byte_index;
                if absolute >= byte_len {
                    break;
                }
                bytes[byte_len - 1 - absolute] = (limb >> (byte_index * 8)) as u8;
            }
        }
        Ok(bytes)
    }

    fn normalize(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn bit(&self, index: usize) -> bool {
        let limb = index / 64;
        let offset = index % 64;
        self.limbs
            .get(limb)
            .is_some_and(|value| ((value >> offset) & 1) != 0)
    }

    fn is_power_of_two(&self) -> bool {
        let mut seen = false;
        for limb in &self.limbs {
            if *limb == 0 {
                continue;
            }
            if seen || !limb.is_power_of_two() {
                return false;
            }
            seen = true;
        }
        seen
    }

    fn trailing_zeros(&self) -> usize {
        for (index, limb) in self.limbs.iter().copied().enumerate() {
            if limb != 0 {
                return index * 64 + limb.trailing_zeros() as usize;
            }
        }
        0
    }

    fn sub_assign(&mut self, other: &Self) {
        debug_assert!(&*self >= other);
        let mut borrow = 0_u64;
        for index in 0..self.limbs.len() {
            let right = other.limbs.get(index).copied().unwrap_or(0);
            let (first, first_borrow) = self.limbs[index].overflowing_sub(right);
            let (second, second_borrow) = first.overflowing_sub(borrow);
            self.limbs[index] = second;
            borrow = if first_borrow || second_borrow { 1 } else { 0 };
        }
        debug_assert_eq!(borrow, 0);
        self.normalize();
    }

    fn shr_assign(&mut self, shift: usize) {
        if shift == 0 || self.is_zero() {
            return;
        }
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        if word_shift >= self.limbs.len() {
            self.limbs.clear();
            return;
        }
        self.limbs.drain(0..word_shift);
        if bit_shift != 0 {
            let mut carry = 0_u64;
            for limb in self.limbs.iter_mut().rev() {
                let next_carry = *limb << (64 - bit_shift);
                *limb = (*limb >> bit_shift) | carry;
                carry = next_carry;
            }
        }
        self.normalize();
    }

    fn shl_one_reserved(&mut self) {
        debug_assert!(self.limbs.len() < self.limbs.capacity());
        let mut carry = 0_u64;
        for limb in &mut self.limbs {
            let next = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next;
        }
        if carry != 0 {
            self.limbs.push(carry);
        }
    }

    fn add_one_reserved(&mut self) {
        for limb in &mut self.limbs {
            let (value, carry) = limb.overflowing_add(1);
            *limb = value;
            if !carry {
                return;
            }
        }
        debug_assert!(self.limbs.len() < self.limbs.capacity());
        self.limbs.push(1);
    }

    fn set_bit_reserved(&mut self, index: usize) {
        let limb = index / 64;
        self.limbs[limb] |= 1_u64 << (index % 64);
    }
}

impl Ord for BigNat {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => self
                .limbs
                .iter()
                .rev()
                .cmp(other.limbs.iter().rev()),
            ordering => ordering,
        }
    }
}

impl PartialOrd for BigNat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for BigNat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return formatter.write_str("0x0");
        }
        formatter.write_str("0x")?;
        for (index, limb) in self.limbs.iter().rev().enumerate() {
            if index == 0 {
                write!(formatter, "{limb:x}")?;
            } else {
                write!(formatter, "{limb:016x}")?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Sign {
    Negative,
    Zero,
    Positive,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BigInt {
    sign: Sign,
    magnitude: BigNat,
}

impl BigInt {
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            sign: Sign::Zero,
            magnitude: BigNat::zero(),
        }
    }

    pub fn try_from_i64(value: i64) -> Result<Self, ArithmeticError> {
        if value == 0 {
            return Ok(Self::zero());
        }
        let sign = if value < 0 {
            Sign::Negative
        } else {
            Sign::Positive
        };
        Ok(Self {
            sign,
            magnitude: BigNat::try_from_u64(value.unsigned_abs())?,
        })
    }

    pub fn from_parts(sign: Sign, magnitude: BigNat) -> Result<Self, ArithmeticError> {
        let canonical = if magnitude.is_zero() {
            Sign::Zero
        } else if sign == Sign::Zero {
            return Err(ArithmeticError::InvalidRational);
        } else {
            sign
        };
        Ok(Self {
            sign: canonical,
            magnitude,
        })
    }

    #[must_use]
    pub const fn sign(&self) -> Sign {
        self.sign
    }

    #[must_use]
    pub const fn magnitude(&self) -> &BigNat {
        &self.magnitude
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.sign == Sign::Zero
    }

    pub fn try_clone(&self) -> Result<Self, ArithmeticError> {
        Ok(Self {
            sign: self.sign,
            magnitude: self.magnitude.try_clone()?,
        })
    }

    pub fn try_negated(&self) -> Result<Self, ArithmeticError> {
        let sign = match self.sign {
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
            Sign::Positive => Sign::Negative,
        };
        Ok(Self {
            sign,
            magnitude: self.magnitude.try_clone()?,
        })
    }

    pub fn try_add(&self, other: &Self) -> Result<Self, ArithmeticError> {
        match (self.sign, other.sign) {
            (Sign::Zero, _) => other.try_clone(),
            (_, Sign::Zero) => self.try_clone(),
            (left, right) if left == right => Self::from_parts(
                left,
                self.magnitude.try_add(&other.magnitude)?,
            ),
            _ => match self.magnitude.cmp(&other.magnitude) {
                Ordering::Equal => Ok(Self::zero()),
                Ordering::Greater => Self::from_parts(
                    self.sign,
                    self.magnitude.try_sub(&other.magnitude)?,
                ),
                Ordering::Less => Self::from_parts(
                    other.sign,
                    other.magnitude.try_sub(&self.magnitude)?,
                ),
            },
        }
    }

    pub fn try_sub(&self, other: &Self) -> Result<Self, ArithmeticError> {
        self.try_add(&other.try_negated()?)
    }

    pub fn try_mul(&self, other: &Self) -> Result<Self, ArithmeticError> {
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero());
        }
        let sign = if self.sign == other.sign {
            Sign::Positive
        } else {
            Sign::Negative
        };
        Self::from_parts(sign, self.magnitude.try_mul(&other.magnitude)?)
    }

    pub fn try_div_exact_nat(&self, divisor: &BigNat) -> Result<Self, ArithmeticError> {
        if self.is_zero() {
            return Ok(Self::zero());
        }
        Self::from_parts(self.sign, self.magnitude.try_div_exact(divisor)?)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GaussianRational {
    real: BigInt,
    imaginary: BigInt,
    denominator: BigNat,
}

impl GaussianRational {
    pub fn try_new(
        real: BigInt,
        imaginary: BigInt,
        denominator: BigNat,
    ) -> Result<Self, ArithmeticError> {
        if denominator.is_zero() {
            return Err(ArithmeticError::DivisionByZero);
        }
        let numerator_gcd = real.magnitude().try_gcd(imaginary.magnitude())?;
        let divisor = numerator_gcd.try_gcd(&denominator)?;
        let (real, imaginary, denominator) = if divisor.is_one() {
            (real, imaginary, denominator)
        } else {
            (
                real.try_div_exact_nat(&divisor)?,
                imaginary.try_div_exact_nat(&divisor)?,
                denominator.try_div_exact(&divisor)?,
            )
        };
        Ok(Self {
            real,
            imaginary,
            denominator,
        })
    }

    pub fn try_zero() -> Result<Self, ArithmeticError> {
        Self::try_new(
            BigInt::zero(),
            BigInt::zero(),
            BigNat::try_from_u64(1)?,
        )
    }

    pub fn try_one() -> Result<Self, ArithmeticError> {
        Self::try_new(
            BigInt::try_from_i64(1)?,
            BigInt::zero(),
            BigNat::try_from_u64(1)?,
        )
    }

    pub fn try_i() -> Result<Self, ArithmeticError> {
        Self::try_new(
            BigInt::zero(),
            BigInt::try_from_i64(1)?,
            BigNat::try_from_u64(1)?,
        )
    }

    #[must_use]
    pub const fn real(&self) -> &BigInt {
        &self.real
    }

    #[must_use]
    pub const fn imaginary(&self) -> &BigInt {
        &self.imaginary
    }

    #[must_use]
    pub const fn denominator(&self) -> &BigNat {
        &self.denominator
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imaginary.is_zero()
    }

    pub fn try_clone(&self) -> Result<Self, ArithmeticError> {
        Ok(Self {
            real: self.real.try_clone()?,
            imaginary: self.imaginary.try_clone()?,
            denominator: self.denominator.try_clone()?,
        })
    }

    pub fn try_mul(&self, other: &Self) -> Result<Self, ArithmeticError> {
        let ac = self.real.try_mul(&other.real)?;
        let bd = self.imaginary.try_mul(&other.imaginary)?;
        let ad = self.real.try_mul(&other.imaginary)?;
        let bc = self.imaginary.try_mul(&other.real)?;
        Self::try_new(
            ac.try_sub(&bd)?,
            ad.try_add(&bc)?,
            self.denominator.try_mul(&other.denominator)?,
        )
    }

    pub fn try_mul_ratio(
        &self,
        numerator: &BigNat,
        denominator: &BigNat,
    ) -> Result<Self, ArithmeticError> {
        if denominator.is_zero() || numerator.is_zero() {
            if denominator.is_zero() {
                return Err(ArithmeticError::DivisionByZero);
            }
            return Self::try_zero();
        }
        let factor = Self::try_new(
            BigInt::from_parts(Sign::Positive, numerator.try_clone()?)?,
            BigInt::zero(),
            denominator.try_clone()?,
        )?;
        self.try_mul(&factor)
    }

    pub fn try_mul_i(&self) -> Result<Self, ArithmeticError> {
        Ok(Self {
            real: self.imaginary.try_negated()?,
            imaginary: self.real.try_clone()?,
            denominator: self.denominator.try_clone()?,
        })
    }

    pub fn try_rotate_i(&self, turns: u8) -> Result<Self, ArithmeticError> {
        let mut value = self.try_clone()?;
        for _ in 0..(turns & 3) {
            value = value.try_mul_i()?;
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{BigInt, BigNat, GaussianRational, Sign};

    fn nat(value: u64) -> BigNat {
        BigNat::try_from_u64(value).unwrap()
    }

    #[test]
    fn arbitrary_width_add_mul_and_divide() {
        let high = nat(1).try_shl_bits(191).unwrap();
        let left = high.try_add_u64(17).unwrap();
        let right = nat(1).try_shl_bits(129).unwrap().try_add_u64(9).unwrap();
        let product = left.try_mul(&right).unwrap();
        let (quotient, remainder) = product.try_div_rem(&left).unwrap();
        assert_eq!(quotient, right);
        assert!(remainder.is_zero());
    }

    #[test]
    fn binary_gcd_and_exact_division_are_canonical() {
        let left = nat(84).try_shl_bits(130).unwrap();
        let right = nat(126).try_shl_bits(128).unwrap();
        let gcd = left.try_gcd(&right).unwrap();
        assert_eq!(gcd, nat(42).try_shl_bits(128).unwrap());
        assert_eq!(left.try_div_exact(&gcd).unwrap(), nat(8));
    }

    #[test]
    fn gaussian_rational_normalizes_and_rotates_exactly() {
        let value = GaussianRational::try_new(
            BigInt::from_parts(Sign::Positive, nat(6)).unwrap(),
            BigInt::from_parts(Sign::Negative, nat(12)).unwrap(),
            nat(18),
        )
        .unwrap();
        assert_eq!(value.real().magnitude(), &nat(1));
        assert_eq!(value.imaginary().magnitude(), &nat(2));
        assert_eq!(value.denominator(), &nat(3));

        let rotated = value.try_mul_i().unwrap();
        assert_eq!(rotated.real().sign(), Sign::Positive);
        assert_eq!(rotated.real().magnitude(), &nat(2));
        assert_eq!(rotated.imaginary().sign(), Sign::Positive);
        assert_eq!(rotated.imaginary().magnitude(), &nat(1));
        assert_eq!(rotated.denominator(), &nat(3));
    }

    #[test]
    fn big_endian_encoding_has_no_layout_leakage() {
        let value = BigNat::try_from_be_bytes(&[0, 0, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(value.try_to_be_bytes().unwrap(), [1, 2, 3, 4, 5]);
    }
}
