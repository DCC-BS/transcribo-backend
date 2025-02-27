## Python Coding Standards for Copilot

1. Type Annotations:
   - Always use type annotations compatible with Python 3.10 and above.
   - Utilize new features like Union | for union types.

2. Ruff Configuration:
   - Adhere to the following Ruff settings:
     - Use specified cache-dir
     - Apply 'extend' rules
     - Follow the defined output-format
     - Apply fixes (including unsafe fixes) as per 'fix' and 'unsafe-fixes'
     - Respect 'fix-only' and 'show-fixes' settings
     - Ensure compatibility with 'required-version'
     - Apply 'preview' features if enabled
     - Follow 'exclude', 'extend-exclude', 'extend-include', and 'force-exclude' patterns
     - Include specified files/directories as per 'include'
     - Respect .gitignore as defined in 'respect-gitignore'
     - Consider specified 'builtins'
     - Handle 'namespace-packages' appropriately
     - Target specified Python version(s) in 'target-version'
     - Use defined 'src' directory
     - Adhere to specified 'line-length' and 'indent-width'
     - Apply 'lint', 'format', and 'analyze' rules

3. Code Documentation:
   - Update docstrings and comments when modifying existing code.
   - Ensure all new functions, classes, and modules have appropriate docstrings.

4. Code Optimization:
   - Suggest simplifications or refactorings that improve readability.
   - Recommend performance optimizations where applicable.
   - Highlight potential areas for code restructuring to enhance maintainability.

5. Best Practices:
   - Follow PEP 8 style guide for Python code.
   - Use f-strings for string formatting when appropriate.
   - Prefer list comprehensions over map() or filter() where it enhances readability.
   - Utilize context managers (with statements) for resource management.

Always strive for clean, efficient, and well-documented code that aligns with modern Python practices and the specified Ruff configuration.
