-- How a tenant connects to a rental (SSH box vs HTTP endpoint). The agent
-- reports it on the `running` report; clients render the matching hint.
ALTER TABLE rentals ADD COLUMN kind TEXT NOT NULL DEFAULT 'ssh';
