# Good and Bad Tests

## Good Tests

**Minimal unit tests**: Verify one important invariant using in-memory inputs and no external effects.

```typescript
// GOOD: One isolated invariant
test("discount never makes a total negative", () => {
  expect(applyDiscount(5, 10)).toBe(0);
});
```

Characteristics:

- Tests behavior users/callers care about
- Uses in-memory inputs
- Has no filesystem, network, subprocess, plotting, or service setup
- Survives internal refactors
- Describes WHAT, not HOW
- Covers only the highest-risk behavior

## Bad Tests

**Implementation-detail tests**: Coupled to internal structure.

```typescript
// BAD: Tests implementation details
test("checkout calls paymentService.process", async () => {
  const mockPayment = jest.mock(paymentService);
  await checkout(cart, payment);
  expect(mockPayment.process).toHaveBeenCalledWith(cart.total);
});
```

Red flags:

- Integration or end-to-end setup
- Filesystem, network, subprocess, plotting, or artifact output
- Large fixtures and broad scenario matrices
- Mocking internal collaborators
- Testing private methods
- Asserting on call counts/order
- Test breaks when refactoring without behavior change
- Test name describes HOW not WHAT
- Verifying through external means instead of interface
