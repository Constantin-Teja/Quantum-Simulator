# Quantum-Simulator

Prerequisites:

- Python >3

## Submodules

This git repository has two submodules:

- https://github.com/KetBy/KetBy-app on path `/app`
- https://github.com/KetBy/KetBy-API on path `/api`

Please make sure to checkout to branch `migration-to-droplet` on the `api` submodule. Afterwards, install the two modules according to their respective documentations.

### API

Prerequisites:

- PHP >=8.3
- Composer
- MySQL server
- Recaptcha

Create the `.env` file following the `.env.example` file, then run:

- `composer install` to install all PHP dependencies
- `php artisan migrate` to migrate the database tables
- `php artisan serve` to start a Laravel development server

### App

Prerequisites:

- Node.js
- npm

Create the `.env.local` file with the following keys:

- `NEXT_PUBLIC_API_URL` with the URL of the API server index
- `NEXT_PUBLIC_RECAPTCHA_SITE_KEY`
- `NEXT_PUBLIC_APP_URL` with value `http://localhost:3000`

Run `npm i` to install all dependencies, then run `npm run dev` to start a NextJS development server on `http://localhost:3000`.

### Putting it all together

The `api` submodule uses the Python scripts in subfolder `/qiskit` to run quantum simulations using the Qiskit library. More precisely, the API submodule invoques the `/qiskit/get_info.py` script to handle the quantum logic.

In order to use the quantum implementation of this project, you must create two symlinks:

- from `/api/qiskit/get_info.py` to `/get_info.py`
- from `/api/qiskit/utils.py` to `/utils.py`

For instance, on Unix-compliant operating systems, one can create a symlink using the following command:

```
$ cd /base/repo/directory
$ ln s /api/qiskit/get_info.py /get_info.py
```
